#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:10:44 2017
@author: wroscoe
remotes.py
The client and web server needed to control a car remotely.
"""

from ...config import Config

import os
import json
import logging
import time
import asyncio

import requests
from tornado.ioloop import IOLoop
from tornado.web import Application, RedirectHandler, StaticFileHandler, \
    RequestHandler
from tornado.httpserver import HTTPServer
import tornado.gen
import tornado.websocket
from socket import gethostname
from donkeycar.surface_handler import compute_throttle, compute_steering_angle
from ... import utils

logger = logging.getLogger(__name__)
logging.getLogger('tornado.access').propagate = False

class RemoteWebServer():
    '''
    A controller that repeatedly polls a remote webserver and expects
    the response to be angle, throttle and drive mode.
    '''

    def __init__(self, remote_url, connection_timeout=.25):

        self.control_url = remote_url
        self.time = 0.
        self.angle = 0.
        self.throttle = 0.
        self.mode = 'user'
        self.mode_latch = None
        self.recording = False
        # use one session for all requests
        self.session = requests.Session()

    def update(self):
        '''
        Loop to run in separate thread the updates angle, throttle and
        drive mode.
        '''

        while True:
            # get latest value from server
            self.angle, self.throttle, self.mode, self.recording = self.run()


    def run_threaded(self):
        '''
        Return the last state given from the remote server.
        '''

        return self.angle, self.throttle, self.mode, self.recording

    def run(self):
        '''
        Posts current car sensor data to webserver and returns
        angle and throttle recommendations.
        '''

        data = {}
        response = None
        while response is None:
            try:
                response = self.session.post(self.control_url,
                                             files={'json': json.dumps(data)},
                                             timeout=0.25)

            except requests.exceptions.ReadTimeout as err:
                print("\n Request took too long. Retrying")
                # Lower throttle to prevent runaways.
                return self.angle, self.throttle * .8, None

            except requests.ConnectionError as err:
                # try to reconnect every 3 seconds
                print("\n Vehicle could not connect to server. Make sure you've " +
                    "started your server and you're referencing the right port.")
                time.sleep(3)

        data = json.loads(response.text)
        angle = float(data['angle'])
        throttle = float(data['throttle'])
        drive_mode = str(data['drive_mode'])
        recording = bool(data['recording'])


        return angle, throttle, drive_mode, recording

    def shutdown(self):
        pass


class LocalWebController(tornado.web.Application):

    def __init__(self, port=8887, mode='user', cfg: Config = None, basic_ctr=None):
        """
        Create and publish variables needed on many of
        the web handlers.
        """
        logger.info('Starting Donkey Server...')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        self.template_path = os.path.join(this_dir, 'templates')
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = mode
        self.mode_latch = None
        self.recording = False
        self.recording_latch = None
        self.buttons = {}  # latched button values for processing
        self.basic_ctr = basic_ctr


        self.port = port

        self.circuit = "Default"
        self.surface = "Dry"
        self.circuit_icon = "/static/images/default_circuit.png"  # For blob image data

        self.num_records = 0
        self.wsclients = []
        self.loop = None
        self.cfg = cfg
        if self.cfg is not None:
            # Pass the config object for altering AI_THROTTLE_MULT
            self.DEFAULT_AI_THROTTLE_MULT = self.cfg.AI_THROTTLE_MULT
        
        # Initialize throttle-related attributes on the application
        self.max_throttle = 1.0
        self.throttle_mode = "default"
        self.straight_throttle = 1.0
        self.steer_throttle = 1.0
        self.basic_ctr.socket_update_fn = self.send_websocket_data

        handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI, dict(ai_throttle_mul=self.cfg.AI_THROTTLE_MULT if self.cfg is not None else 0.0)),
            (r"/dashboard", DashboardAPI),
            (r"/wsDrive", WebSocketDriveAPI, dict(cfg=self.cfg)),
            (r"/api/drive", DrivePostAPI),  # Separate endpoint for POST requests from joystick
            (r"/wsCalibrate", WebSocketCalibrateAPI),
            (r"/calibrate", CalibrateHandler),
            (r"/video", VideoAPI),
            (r"/wsTest", WsTest),
            (r"/circuit", CircuitAPI, dict(circuit=self.circuit, surface=self.surface, circuit_icon=self.circuit_icon)),

            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path}),
        ]

        settings = {'debug': True}
        super().__init__(handlers, **settings)
        logger.info(f"You can now go to {gethostname()}.local:{port} to "
                    f"drive your car.")

    def update(self):
        """ Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        self.loop = IOLoop.instance()
        self.loop.start()

    def update_wsclients(self, data):
        if data:
            for wsclient in self.wsclients:
                try:
                    data_str = json.dumps(data)
                    logger.debug(f"Updating web client: {data_str}")
                    wsclient.write_message(data_str)
                except Exception as e:
                    logger.warning("Error writing websocket message",
                                   exc_info=e)
                    pass

    def send_websocket_data(self, data):
        """Send data directly to WebSocket clients"""

        if self.loop is not None:
            self.loop.add_callback(lambda: self.update_wsclients(data))

    def run_threaded(self, img_arr=None, num_records=0, mode=None, recording=None, custom_values=None, text_content=None):
        """
        :param img_arr: current camera image or None
        :param num_records: current number of data records
        :param mode: default user/mode
        :param recording: default recording mode
        :param custom_values: custom values to display
        :param text_content: text content to display
        """
        self.img_arr = img_arr
        self.num_records = num_records

        #
        # enforce defaults if they are not none.
        #
        changes = {}
        if mode is not None and self.mode != mode:
            self.mode = mode
            changes["driveMode"] = self.mode
        if self.mode_latch is not None:
            self.mode = self.mode_latch
            self.mode_latch = None
            changes["driveMode"] = self.mode
        if recording is not None and self.recording != recording:
            self.recording = recording
            changes["recording"] = self.recording
        if self.recording_latch is not None:
            self.recording = self.recording_latch;
            self.recording_latch = None;
            changes["recording"] = self.recording;

        # Send record count to websocket clients
        if (self.num_records is not None and self.recording is True):
            if self.num_records % 10 == 0:
                changes['num_records'] = self.num_records

        # Send custom values if provided
        if custom_values is not None:
            changes['custom_values'] = custom_values

        # Send text content if provided
        if text_content is not None:
            changes['text_content'] = text_content

        #
        # get latched button presses then clear button presses
        # Next iteration will clear press in memory
        #
        buttons = self.buttons
        self.buttons = {}
        for button, pressed in buttons.items():
            if pressed:
                self.buttons[button] = False
        # if there were changes, or if we have custom data to send, then send to web client
        if (changes or custom_values is not None or text_content is not None) and self.loop is not None:
            self.loop.add_callback(lambda: self.update_wsclients(changes))

        return self.angle, self.throttle, self.mode, self.recording, buttons

    def run(self, img_arr=None, num_records=0, mode=None, recording=None, custom_values=None, text_content=None):
        return self.run_threaded(img_arr, num_records, mode, recording, custom_values, text_content)

    def shutdown(self):
        pass


class DriveAPI(RequestHandler):


    def initialize(self, ai_throttle_mul: float = 0.0) -> None:
        self.ai_throttle_mul = ai_throttle_mul


    def get(self):
        data = {
            "current_ai_mul": str(self.ai_throttle_mul),
            "current_circuit": self.application.circuit,
            "current_surface": self.application.surface,
            "current_circuit_icon": self.application.circuit_icon
        }
        self.render("templates/vehicle.html", **data)

    def post(self):
        '''
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        '''
        data = tornado.escape.json_decode(self.request.body)

        if data.get('angle') is not None:
            self.application.angle = data['angle']
        if data.get('throttle') is not None:
            self.application.throttle = data['throttle']
        if data.get('drive_mode') is not None:
            self.application.mode = data['drive_mode']
        if data.get('recording') is not None:
            self.application.recording = data['recording']
        if data.get('buttons') is not None:
            latch_buttons(self.application.buttons, data['buttons'])


class WsTest(RequestHandler):
    def get(self):
        data = {}
        self.render("templates/wsTest.html", **data)


class CalibrateHandler(RequestHandler):
    """ Serves the calibration web page"""
    async def get(self):
        await self.render("templates/calibrate.html")


def latch_buttons(buttons, pushes):
    """
    Latch button pushes
    buttons: the latched values
    pushes: the update value
    """
    if pushes is not None:
        #
        # we got button pushes.
        # - we latch the pushed buttons so we can process the push
        # - after it is processed we clear it
        #
        for button in pushes:
            # if pushed, then latch it
            if pushes[button]:
                buttons[button] = True


class DrivePostAPI(RequestHandler):
    '''
    Handles POST requests from the joystick controller.
    Separated from WebSocket handler to ensure proper application instance sharing.
    '''
    
    def limited_throttle(
        self,
        new_throttle,
        max_throttle,
        throttle_mode,
        straight_throttle,
        steer_throttle,
        steer_angle
    ):
        limited_throttle = 0

        if new_throttle > 0:
            limited_throttle = min(max_throttle, new_throttle)

        if new_throttle < 0:
            limited_throttle = max(-max_throttle, new_throttle)

        if throttle_mode == "constant":
            limited_throttle = max_throttle

        if throttle_mode == "steer_limited":
            # Interpolate between straight throttle and full steer throttle
            steer_amount = abs(steer_angle)  # 0 to 1
            max_allowed_throttle = (
                straight_throttle +
                (steer_throttle - straight_throttle) * steer_amount
            )

            if new_throttle > 0:
                limited_throttle = min(max_allowed_throttle, new_throttle)
            elif new_throttle < 0:
                limited_throttle = max(-max_allowed_throttle, new_throttle)

        return limited_throttle

    def post(self):
        '''
        Receive post requests from joystick controller
        '''
        data = tornado.escape.json_decode(self.request.body)
        angle = data["angle"]
        throttle = data["throttle"]
        
        # Get throttle parameters from application
        max_throttle = self.application.max_throttle
        throttle_mode = self.application.throttle_mode
        straight_throttle = self.application.straight_throttle
        steer_throttle = self.application.steer_throttle
        
        throttle = self.limited_throttle(
            throttle,
            max_throttle,
            throttle_mode,
            straight_throttle,
            steer_throttle,
            angle
        )
        new_throttle = compute_throttle(throttle, self.application.throttle, self.application.surface)
        new_steering = compute_steering_angle(angle, throttle, self.application.angle, self.application.surface)
        self.application.angle = new_steering
        self.application.throttle = float(new_throttle) if abs(float(new_throttle)) > 0.1 else 0

        angle_to_print = float(new_steering) if abs(float(new_steering)) > 0.05 else 0
        self.write({"angle": new_steering, "throttle": self.application.throttle})
        changes = {"throttle": self.application.throttle , "angle": angle_to_print}
        self.application.send_websocket_data(changes)



class WebSocketDriveAPI(tornado.websocket.WebSocketHandler):
    def initialize(self, cfg: Config) -> None:
        self.cfg = cfg
    
    def check_origin(self, origin):
        return True

    def open(self):
        logger.info(f"New WebSocket client connected - app_id: {id(self.application)}")
        self.application.wsclients.append(self)

    def on_message(self, message):
        data = json.loads(message)
        # logger.info(f"WebSocket received from drive page: {message}")
        self.application.surface = data.get('surface', self.application.surface)
        self.application.max_throttle = data.get('max_throttle', self.application.max_throttle)
        self.application.throttle_mode = data.get('throttle_mode', self.application.throttle_mode)
        self.application.straight_throttle = data.get('straight_throttle', self.application.straight_throttle)
        self.application.steer_throttle = data.get('steer_throttle', self.application.steer_throttle)
        
        # logger.info(f"Updated application attributes - max_throttle: {self.application.max_throttle}, throttle_mode: {self.application.throttle_mode}, app_id: {id(self.application)}")
        
        new_throttle = compute_throttle(data.get('throttle', self.application.throttle), self.application.throttle, self.application.surface)
        new_steering = compute_steering_angle(data.get('angle', self.application.angle), data.get('throttle', self.application.throttle), self.application.angle, self.application.surface)
        self.application.angle = new_steering
        self.application.throttle = new_throttle
        
        # logger.info(f"Computed values - angle: {new_steering}, throttle: {new_throttle}")
       
        changes = {}
        
        # Send angle and throttle updates to all connected clients in nested structure
        if 'angle' in data or 'throttle' in data:
            changes['tele'] = {'user': {}}
            if 'angle' in data:
                changes['tele']['user']['angle'] = new_steering
            if 'throttle' in data:
                changes['tele']['user']['throttle'] = new_throttle
        
        # Update circuit and track changes
        if data.get('circuit') is not None and self.application.circuit != data['circuit']:
            self.application.circuit = data['circuit']
            changes['circuit'] = self.application.circuit
            
        # Update circuit_icon and track changes  
        if data.get('circuit_icon') is not None and self.application.circuit_icon != data['circuit_icon']:
            self.application.circuit_icon = data['circuit_icon']
            changes['circuit_icon'] = self.application.circuit_icon
            
        # Update surface and track changes
        if data.get('surface') is not None and self.application.surface != data['surface']:
            self.application.surface = data['surface']
            changes['surface'] = self.application.surface

        if data.get('drive_mode') is not None:
            self.application.mode = data['drive_mode']
            self.application.mode_latch = self.application.mode
        if data.get('recording') is not None:
            self.application.recording = data['recording']
            self.application.recording_latch = self.application.recording
        if data.get('buttons') is not None:
            latch_buttons(self.application.buttons, data['buttons'])
        if data.get('ai_throttle_update') is not None:
            self.cfg.AI_THROTTLE_MULT = float(data['ai_throttle_update'])
            
        # Send updates to all WebSocket clients if there were changes
        if changes:
            logger.debug(f"Broadcasting changes to clients: {changes}")
            self.application.send_websocket_data(changes)
        if(self.application.basic_ctr is not None):
            self.application.basic_ctr.max_throttle = self.application.max_throttle
            self.application.basic_ctr.surface = self.application.surface
            self.application.basic_ctr.throttle_mode = self.application.throttle_mode
            self.application.basic_ctr.straight_throttle = self.application.straight_throttle
            self.application.basic_ctr.steer_throttle = self.application.steer_throttle

    def on_close(self):
        logger.info("Client disconnected")
        self.application.wsclients.remove(self)



class WebSocketCalibrateAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        logger.info("New client connected")

    def on_message(self, message):
        logger.info(f"wsCalibrate {message}")
        data = json.loads(message)
        if 'throttle' in data:
            print(data['throttle'])
            self.application.throttle = data['throttle']

        if 'angle' in data:
            print(data['angle'])
            self.application.angle = data['angle']

        if 'config' in data:
            config = data['config']
            if self.application.drive_train_type == "PWM_STEERING_THROTTLE" \
                or self.application.drive_train_type == "I2C_SERVO":
                if 'STEERING_LEFT_PWM' in config:
                    self.application.drive_train['steering'].left_pulse = config['STEERING_LEFT_PWM']

                if 'STEERING_RIGHT_PWM' in config:
                    self.application.drive_train['steering'].right_pulse = config['STEERING_RIGHT_PWM']

                if 'THROTTLE_FORWARD_PWM' in config:
                    self.application.drive_train['throttle'].max_pulse = config['THROTTLE_FORWARD_PWM']

                if 'THROTTLE_STOPPED_PWM' in config:
                    self.application.drive_train['throttle'].zero_pulse = config['THROTTLE_STOPPED_PWM']

                if 'THROTTLE_REVERSE_PWM' in config:
                    self.application.drive_train['throttle'].min_pulse = config['THROTTLE_REVERSE_PWM']

            elif self.application.drive_train_type == "MM1":
                if ('MM1_STEERING_MID' in config) and (config['MM1_STEERING_MID'] != 0):
                        self.application.drive_train.STEERING_MID = config['MM1_STEERING_MID']
                if ('MM1_MAX_FORWARD' in config) and (config['MM1_MAX_FORWARD'] != 0):
                        self.application.drive_train.MAX_FORWARD = config['MM1_MAX_FORWARD']
                if ('MM1_MAX_REVERSE' in config) and (config['MM1_MAX_REVERSE'] != 0):
                    self.application.drive_train.MAX_REVERSE = config['MM1_MAX_REVERSE']

    def on_close(self):
        logger.info("Client disconnected")


class VideoAPI(RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle.
    '''

    async def get(self):
        placeholder_image = utils.load_image_sized(
                        os.path.join(self.application.static_file_path,
                                     "img_placeholder.jpg"), 160, 120, 3)

        self.set_header("Content-type",
                        "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        
        last_img_id = None
        
        while True:
            # Reduce interval to 20 FPS (instead of 200 FPS)
            interval = .05
            
            if served_image_timestamp + interval < time.time():
            
                current_img_arr = getattr(self.application, 'img_arr', None)

                if current_img_arr is not None and id(current_img_arr) != last_img_id:
                
                    img = utils.arr_to_binary(current_img_arr)
                    last_img_id = id(current_img_arr)

                    self.write(my_boundary)
                    self.write("Content-type: image/jpeg\r\n")
                    self.write("Content-length: %s\r\n\r\n" % len(img))
                    self.write(img)
                    served_image_timestamp = time.time()
                    try:
                        await self.flush()
                    except tornado.iostream.StreamClosedError:
                        break 
                elif current_img_arr is None:
                    img = utils.arr_to_binary(placeholder_image)
                    self.write(my_boundary)
                    self.write("Content-type: image/jpeg\r\n")
                    self.write("Content-length: %s\r\n\r\n" % len(img))
                    self.write(img)
                    served_image_timestamp = time.time()
                    try:
                        await self.flush()
                    except tornado.iostream.StreamClosedError:
                        break

            await tornado.gen.sleep(0.005)


class BaseHandler(RequestHandler):
    """ Serves the FPV web page"""
    async def get(self):
        data = {}
        await self.render("templates/base_fpv.html", **data)


class WebFpv(Application):
    """
    Class for running an FPV web server that only shows the camera in real-time.
    The web page contains the camera view and auto-adjusts to the web browser
    window size. Conjecture: this picture up-scaling is performed by the
    client OS using graphics acceleration. Hence a web browser on the PC is
    faster than a pure python application based on open cv or similar.
    """

    def __init__(self, port=8890):
        self.port = port
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')

        """Construct and serve the tornado application."""
        handlers = [
            (r"/", BaseHandler),
            (r"/video", VideoAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path})
        ]

        settings = {'debug': True}
        self.img_arr = None
        super().__init__(handlers, **settings)
        logger.info(f"Started Web FPV server. You can now go to "
                    f"{gethostname()}.local:{self.port} to view the car camera")

    def update(self):
        """ Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        IOLoop.instance().start()

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr

    def run(self, img_arr=None):
        self.img_arr = img_arr

    def shutdown(self):
        pass


class DashboardAPI(RequestHandler):
    """Serves the dashboard web page using vehicle_show.html"""
    
    def get(self):
        # Map surface to icon path
        surface_icons = {
            "Dry": "/static/weather/dry.png",
            "Wet": "/static/weather/wet.png",
            "Icy": "/static/weather/icy.png"
        }
        current_surface_icon = surface_icons.get(self.application.surface, "/static/weather/dry.png")
        angle_to_print = float(self.application.throttle) if abs(float(self.application.throttle)) > 0.05 else 0
        data = {
            "current_ai_mul": str(self.application.cfg.AI_THROTTLE_MULT if self.application.cfg is not None else 0.0),
            "current_circuit": self.application.circuit,
            "current_surface": self.application.surface,
            "current_surface_icon": current_surface_icon,
            "current_circuit_icon": self.application.circuit_icon,
            "throttle": self.application.throttle,
            "angle": angle_to_print
        }
        self.render("templates/vehicle_show.html", **data)


class CircuitAPI(RequestHandler):

    def initialize(self, 
                   circuit: str = "Default",
                   surface: str = "Dry",
                   circuit_icon: str = "static/images/circuit_icon.png") -> None:
        self.circuit = circuit
        self.surface = surface
        self.circuit_icon = circuit_icon

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    
    def options(self):
        # no body
        self.set_status(204)
        self.finish()

    def get(self):
        data = {"current_circuit": self.circuit, "current_surface": self.surface, "circuit_icon": self.circuit_icon}
        self.render("templates/vehicle.html", **data)

    def post(self):
        '''
        Receive post requests as user changes the circuit
        and surface of the vehicle on the webpage
        '''
        data = tornado.escape.json_decode(self.request.body)
        
        changes = {}
        if data.get('circuit') is not None:
            self.application.circuit = data['circuit']
            changes['circuit'] = self.application.circuit
        if data.get('surface') is not None:
            self.application.surface = data['surface']
            changes['surface'] = self.application.surface
            # Also send the updated surface icon
            surface_icons = {
                "Dry": "/static/weather/dry.png",
                "Wet": "/static/weather/wet.png",
                "Icy": "/static/weather/icy.png"
            }
            changes['surface_icon'] = surface_icons.get(self.application.surface, "/static/weather/dry.png")
        if data.get('circuit_icon') is not None:
            self.application.circuit_icon = data['circuit_icon']
            changes['circuit_icon'] = self.application.circuit_icon
            
        # Send updates to WebSocket clients
        if changes:
            logger.info(f"CircuitAPI broadcasting changes: {changes}")
            self.application.send_websocket_data(changes)
        if(self.application.basic_ctr is not None):
            self.application.basic_ctr.surface = self.application.surface