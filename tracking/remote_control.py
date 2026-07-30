#!/usr/bin/env python3

import asyncio
import json

import websockets

URI = "ws://piracerpro-8.local:8887/wsDrive"

SEND_INTERVAL_SECONDS = 1.0
RECONNECT_DELAY_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

# throttle:
#   positive = forward
#   negative = reverse, if supported by the vehicle
#
# angle:
#   -1.0 = full left
#    0.0 = straight
#    1.0 = full right

COMMANDS = [
    {"throttle": 0.20, "angle": 1.0},
    {"throttle": 0.20, "angle": 0.5},
    {"throttle": 0.30, "angle": 0.0},
    {"throttle": 0.40, "angle": -0.2},
    {"throttle": 0.20, "angle": -0.3},
    {"throttle": 0.40, "angle": 0.0},
]

def validate_command(command):
    """
    Validate and normalize one command.
    """

    if "throttle" not in command or "angle" not in command:
        raise ValueError(
            "Each command must contain 'throttle' and 'angle'."
        )

    throttle = float(command["throttle"])
    angle = float(command["angle"])

    # Keep steering within the expected range.
    angle = max(-0.99, min(0.99, angle))

    return {
        "throttle": throttle,
        "angle": angle,
    }


async def send_command(websocket, command):
    """
    Send one command through the WebSocket.
    """

    command = validate_command(command)
    message = json.dumps(command)

    await websocket.send(message)

    print(
        f"Sent command: "
        f"throttle={command['throttle']:.2f}, "
        f"angle={command['angle']:.2f}"
    )


async def run_command_sequence():
    """
    Connect to the car and send one command every second.
    """

    while True:
        try:
            print(f"Connecting to {URI}...")

            async with websockets.connect(URI) as websocket:
                print("Connection established successfully.")

                for command in COMMANDS:
                    await send_command(websocket, command)
                    await asyncio.sleep(SEND_INTERVAL_SECONDS)

                # Always send a stop command after completing the sequence.
                print("Command sequence completed. Stopping the car.")

                await send_command(
                    websocket,
                    {
                        "throttle": 0.0,
                        "angle": 0.0,
                    },
                )

                return

        except (
            OSError,
            websockets.ConnectionClosed,
        ) as exception:
            print(f"WebSocket error: {exception}")
            print(
                f"Retrying in {RECONNECT_DELAY_SECONDS} seconds..."
            )

            await asyncio.sleep(RECONNECT_DELAY_SECONDS)


async def main():
    try:
        await run_command_sequence()

    except KeyboardInterrupt:
        print("\nProgram interrupted.")

    finally:
        print("Program finished.")


if __name__ == "__main__":
    asyncio.run(main())