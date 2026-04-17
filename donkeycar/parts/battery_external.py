import logging
import time
import importlib

logger = logging.getLogger(__name__)

class ExternalBatteryReader:
    """Read battery bus voltage directly from INA219 sensor."""

    def __init__(self,
                 poll_interval_s: float = 1.0,
                 i2c_addr: int = 0x42) -> None:
        self.poll_interval_s = max(0.1, float(poll_interval_s))
        self.ina = None
        try:
            ina219_module = importlib.import_module("pidisplay.ina219")
            self.ina = ina219_module.INA219(addr=i2c_addr)
        except Exception as exc:
            logger.warning("Unable to initialize INA219 battery reader: %s", exc)
        self.last_value = None
        self._next_poll_ts = 0.0

    def _read_sensor(self):
        if self.ina is None:
            return None
        return float(self.ina.getBusVoltage_V())

    def run(self):
        now = time.time()
        if now < self._next_poll_ts:
            return self.last_value

        self._next_poll_ts = now + self.poll_interval_s
        try:
            value = self._read_sensor()
            if value is not None:
                self.last_value = value
            return self.last_value
        except Exception as exc:
            logger.warning("Battery sensor read error: %s", exc)

        return self.last_value
