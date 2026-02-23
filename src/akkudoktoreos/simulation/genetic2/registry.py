"""Device registry for managing simulation participants.

The registry is the single source of truth for all devices in a simulation run.
Devices are registered once and retrieved by ID or type throughout the simulation.
"""

from typing import Iterator, Type, TypeVar

from akkudoktoreos.devices.genetic2.base import EnergyDevice

T = TypeVar("T")


class DeviceRegistry:
    """Holds all devices participating in a simulation run.

    Devices are stored by their unique ``device_id``. Multiple instances
    of the same device class are fully supported (e.g. two batteries,
    two PV strings on separate inverters).

    The registry is the single point of device discovery for both the
    simulation engine and the genome assembler — neither needs to know
    the specific set of devices at construction time.

    Example:
        >>> registry = DeviceRegistry()
        >>> registry.register(Battery(params=p1, device_id="battery_main"))
        >>> registry.register(Battery(params=p2, device_id="battery_garage"))
        >>> for bat in registry.all_of_type(Battery):
        ...     print(bat.device_id)
        battery_main
        battery_garage
    """

    def __init__(self) -> None:
        # Insertion-ordered dict preserves registration order, which determines
        # the order devices are iterated — relevant for simulation sequencing.
        self._devices: dict[str, EnergyDevice] = {}

    def register(self, device: EnergyDevice) -> None:
        """Register a device with the registry.

        Args:
            device (EnergyDevice): Device to register. Must have a unique
                ``device_id`` not already present in this registry.

        Raises:
            ValueError: If a device with the same ``device_id`` is already
                registered. Use distinct IDs for multiple instances of the
                same class.
        """
        if device.device_id in self._devices:
            raise ValueError(
                f"Device '{device.device_id}' is already registered. "
                "Each device instance must have a unique device_id."
            )
        self._devices[device.device_id] = device

    def unregister(self, device_id: str) -> None:
        """Remove a device from the registry.

        Args:
            device_id (str): ID of the device to remove.

        Raises:
            KeyError: If no device with this ID is registered.
        """
        if device_id not in self._devices:
            raise KeyError(f"Device '{device_id}' is not registered.")
        del self._devices[device_id]

    def get(self, device_id: str) -> EnergyDevice:
        """Retrieve a device by its ID.

        Args:
            device_id (str): Unique device identifier.

        Returns:
            EnergyDevice: The registered device.

        Raises:
            KeyError: If no device with this ID is registered.
        """
        if device_id not in self._devices:
            raise KeyError(
                f"Device '{device_id}' not found in registry. "
                f"Registered devices: {list(self._devices.keys())}"
            )
        return self._devices[device_id]

    def get_typed(self, device_id: str, device_type: Type[T]) -> T:
        """Retrieve a device by ID with type checking.

        Args:
            device_id (str): Unique device identifier.
            device_type (Type[T]): Expected type of the device.

        Returns:
            T: The registered device cast to the expected type.

        Raises:
            KeyError: If no device with this ID is registered.
            TypeError: If the device is not an instance of ``device_type``.
        """
        device = self.get(device_id)
        if not isinstance(device, device_type):
            raise TypeError(
                f"Device '{device_id}' is {type(device).__name__}, expected {device_type.__name__}."
            )
        return device

    def all_of_type(self, device_type: type[object]) -> Iterator[EnergyDevice]:
        """Iterate over all registered devices of a specific type.

        Yields devices in registration order.

        Args:
            device_type (Type[object]): Type to filter by (includes subclasses).

        Yields:
            EnergyDevice: Each registered device that is an instance of ``device_type``.

        Example:
            >>> for battery in registry.all_of_type(Battery):
            ...     print(battery.current_soc_percentage())
        """
        if not issubclass(device_type, EnergyDevice):
            raise TypeError("device_type must be subclass of EnergyDevice")

        for device in self._devices.values():
            if isinstance(device, device_type):
                yield device

    def all_devices(self) -> Iterator[EnergyDevice]:
        """Iterate over all registered devices in registration order.

        Yields:
            EnergyDevice: Each registered device.
        """
        yield from self._devices.values()

    def reset_all(self) -> None:
        """Reset all registered devices to their initial state.

        Called by the engine before each simulation run.
        """
        for device in self._devices.values():
            device.reset()

    def device_ids(self) -> list[str]:
        """Return all registered device IDs in registration order.

        Returns:
            list[str]: Ordered list of device IDs.
        """
        return list(self._devices.keys())

    def __contains__(self, device_id: str) -> bool:
        """Check if a device ID is registered.

        Args:
            device_id (str): ID to check.

        Returns:
            bool: True if registered.
        """
        return device_id in self._devices

    def __len__(self) -> int:
        return len(self._devices)

    def __repr__(self) -> str:
        ids = list(self._devices.keys())
        return f"DeviceRegistry({ids})"
