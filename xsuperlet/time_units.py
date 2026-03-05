class SecondToUnit:
    def __init__(self, unit: str, start: int | float) -> None:
        self.unit = unit
        self.start = start

        self.time_units = {
            "seconds": self.seconds,
            "second": self.seconds,
            "secs": self.seconds,
            "sec": self.seconds,
            "s": self.seconds,

            "minutes": self.minutes,
            "minute": self.minutes,
            "mins": self.minutes,
            "min": self.minutes,

            "hours": self.hours,
            "hour": self.hours,
            "hrs": self.hours,
            "hr": self.hours,
            "h": self.hours,

            "days": self.days,
            "d": self.days,

            "years": self.years,
            "year": self.years,
            "yrs": self.years,
            "yr": self.years,
            "y": self.years,

            "MJD": self.mjd,
            "XMM MJD": self.xmm_mjd,

            "ks": self.kiloseconds,
            "Ms": self.megaseconds,
            "Gs": self.gigaseconds,
            "ms": self.milliseconds,
            "μs": self.microseconds,
        }

    @staticmethod
    def seconds(time) -> int | float:
        return time

    @staticmethod
    def minutes(time) -> int | float:
        return time / 60

    def hours(self, time) -> int | float:
        return self.minutes(time) / 60

    def days(self, time) -> int | float:
        return self.hours(time) / 24

    def years(self, time) -> int | float:
        return self.days(time) / 365.24219

    @staticmethod
    def kiloseconds(time) -> int | float:
        return time * 1E-3

    @staticmethod
    def megaseconds(time) -> int | float:
        return time * 1E-6

    @staticmethod
    def gigaseconds(time) -> int | float:
        return time * 1E-9

    @staticmethod
    def milliseconds(time) -> int | float:
        return time * 1E+3

    @staticmethod
    def microseconds(time) -> int | float:
        return time * 1E+6

    def mjd(self, time) -> int | float:
        return time + self.start

    def xmm_mjd(self, time) -> int | float:
        return ((time + self.start) / 86400) +  50814.0

    def seconds_to_unit(self, time: int | float) -> int | float:
        """
        Converts time in seconds to the given unit.

        :param time: Time in seconds
        """
        try:
            return self.time_units[self.unit](time)
        except KeyError:
            print(f"Unknown time unit: {self.unit}\nUnit conversion impossible!")
            return time


frequency_dict = {
    "GHz": 1E+9,
    "MHz": 1E+6,
    "kHz": 1E+3,
    "Hz" : 1E+0,
    "mHz": 1E-3,
    "μHz": 1E-6,
    "nHz": 1E-9,
    "pHz": 1E-12
}
