"""Resource arbitration for the simulation engine.

The arbitrator sits between device requests and device simulation:
it takes all requests for a given hour and decides how much each
device actually gets. Keeping arbitration behind an interface means
the engine never hard-codes allocation policy.
"""

from abc import ABC, abstractmethod

from akkudoktoreos.simulation.genetic2.flows import (
    ResourceGrant,
    ResourceRequest,
)


class ArbitratorBase(ABC):
    """Interface for resource arbitration strategies.

    The engine calls ``arbitrate()`` once per hour with all device requests,
    and passes the resulting grants to each device's ``simulate_hour()``.

    Implement this interface to swap in different allocation policies
    (e.g. priority-based, market-based, proportional rationing) without
    touching the engine or any device code.
    """

    @abstractmethod
    def arbitrate(self, requests: list[ResourceRequest]) -> dict[str, ResourceGrant]:
        """Allocate resources across all requesting devices for one hour.

        Args:
            requests (list[ResourceRequest]): All device requests for this hour,
                including both producers (negative values) and consumers
                (positive values).

        Returns:
            dict[str, ResourceGrant]: Mapping of device_id to allocated grant.
                Every device that submitted a request must have an entry.
                Devices not in the input should not appear in the output.
        """


class PriorityArbitrator(ArbitratorBase):
    """Priority-based resource arbitrator.

    Allocation policy:

    1. **Producers** are always dispatched in full — they never compete.
       Their output is summed to form the available supply pool.
    2. **Consumers** are served in priority order (lowest Priority value first).
       Each consumer gets up to its requested amount from the remaining pool.
    3. If a consumer would receive less than its ``min_*`` threshold,
       it is **curtailed entirely** (grant set to zero, ``curtailed=True``).
       This prevents devices from running in an inefficient partial state
       (e.g. a heat pump below minimum viable power).
    4. Any surplus after all consumers are served represents potential
       grid feed-in or wasted energy — the engine handles this.

    Each resource type (AC, DC, heat, cold) is arbitrated independently.
    A device may be curtailed for AC but still receive heat if available.

    Note:
        This arbitrator treats each resource pool as fully fungible within
        its type. It does not model transmission constraints or phase
        imbalance within AC circuits.
    """

    def arbitrate(self, requests: list[ResourceRequest]) -> dict[str, ResourceGrant]:
        """Allocate resources using priority ordering.

        Args:
            requests (list[ResourceRequest]): All device requests for this hour.

        Returns:
            dict[str, ResourceGrant]: Grants for all requesting devices.
        """
        grants: dict[str, ResourceGrant] = {}

        # Accumulate supply from all producers first
        # Producers have negative request values (offering surplus)
        ac_available = sum(-r.ac_power_wh for r in requests if r.ac_power_wh < 0)
        dc_available = sum(-r.dc_power_wh for r in requests if r.dc_power_wh < 0)
        heat_available = sum(-r.heat_wh for r in requests if r.heat_wh < 0)
        cold_available = sum(-r.cold_wh for r in requests if r.cold_wh < 0)

        # Grant producers their full request immediately — they always run
        for req in requests:
            if req.is_producer:
                grants[req.device_id] = ResourceGrant(
                    device_id=req.device_id,
                    ac_power_wh=req.ac_power_wh,
                    dc_power_wh=req.dc_power_wh,
                    heat_wh=req.heat_wh,
                    cold_wh=req.cold_wh,
                )
            elif req.is_idle:
                grants[req.device_id] = ResourceGrant.idle(req.device_id)

        # Serve consumers in priority order
        consumers = sorted(
            [r for r in requests if r.is_consumer],
            key=lambda r: r.priority,
        )

        for req in consumers:
            # Each resource type is curtailed independently.
            # A device curtailed for AC may still receive heat if available.
            ac_grant = 0.0
            dc_grant = 0.0
            heat_grant = 0.0
            cold_grant = 0.0
            ac_curtailed = False
            dc_curtailed = False
            heat_curtailed = False
            cold_curtailed = False

            # AC allocation
            if req.ac_power_wh > 0:
                if ac_available >= req.min_ac_power_wh:
                    ac_grant = min(req.ac_power_wh, ac_available)
                    ac_available -= ac_grant
                else:
                    ac_curtailed = True

            # DC allocation
            if req.dc_power_wh > 0:
                if dc_available >= req.min_dc_power_wh:
                    dc_grant = min(req.dc_power_wh, dc_available)
                    dc_available -= dc_grant
                else:
                    dc_curtailed = True

            # Heat allocation
            if req.heat_wh > 0:
                if heat_available >= req.min_heat_wh:
                    heat_grant = min(req.heat_wh, heat_available)
                    heat_available -= heat_grant
                else:
                    heat_curtailed = True

            # Cold allocation
            if req.cold_wh > 0:
                if cold_available >= req.min_heat_wh:
                    cold_grant = min(req.cold_wh, cold_available)
                    cold_available -= cold_grant
                else:
                    cold_curtailed = True

            # Device is considered curtailed if ANY requested resource was denied.
            # Individual resource grants are zeroed only for their own curtailment.
            any_curtailed = ac_curtailed or dc_curtailed or heat_curtailed or cold_curtailed
            grants[req.device_id] = ResourceGrant(
                device_id=req.device_id,
                ac_power_wh=0.0 if ac_curtailed else ac_grant,
                dc_power_wh=0.0 if dc_curtailed else dc_grant,
                heat_wh=0.0 if heat_curtailed else heat_grant,
                cold_wh=0.0 if cold_curtailed else cold_grant,
                curtailed=any_curtailed,
            )

        return grants


class ProportionalArbitrator(ArbitratorBase):
    """Proportional resource arbitrator.

    When supply is insufficient to meet all demand, each consumer receives
    a proportional share based on its request size, regardless of priority.
    Devices that would fall below their minimum are curtailed, and their
    share is redistributed proportionally among the remaining consumers.

    This is useful for modelling scenarios where no device has strict
    priority over others (e.g. peer-to-peer energy sharing).
    """

    def arbitrate(self, requests: list[ResourceRequest]) -> dict[str, ResourceGrant]:
        """Allocate resources proportionally.

        Args:
            requests (list[ResourceRequest]): All device requests for this hour.

        Returns:
            dict[str, ResourceGrant]: Proportional grants for all devices.
        """
        grants: dict[str, ResourceGrant] = {}

        ac_available = sum(-r.ac_power_wh for r in requests if r.ac_power_wh < 0)
        dc_available = sum(-r.dc_power_wh for r in requests if r.dc_power_wh < 0)
        heat_available = sum(-r.heat_wh for r in requests if r.heat_wh < 0)

        for req in requests:
            if req.is_producer:
                grants[req.device_id] = ResourceGrant(
                    device_id=req.device_id,
                    ac_power_wh=req.ac_power_wh,
                    dc_power_wh=req.dc_power_wh,
                    heat_wh=req.heat_wh,
                )
            elif req.is_idle:
                grants[req.device_id] = ResourceGrant.idle(req.device_id)

        consumers = [r for r in requests if r.is_consumer]

        grants.update(self._proportional_allocate("ac", consumers, ac_available))
        grants.update(self._proportional_allocate("heat", consumers, heat_available))

        return grants

    def _proportional_allocate(
        self,
        resource: str,
        consumers: list[ResourceRequest],
        available: float,
    ) -> dict[str, ResourceGrant]:
        """Allocate a single resource proportionally with curtailment loop.

        Args:
            resource (str): Resource name ("ac", "dc", or "heat").
            consumers (list[ResourceRequest]): Consumer requests.
            available (float): Total supply available.

        Returns:
            dict[str, ResourceGrant]: Partial grants for this resource.
        """
        attr = f"{resource}_power_wh" if resource in ("ac", "dc") else "heat_wh"
        min_attr = f"min_{resource}_power_wh" if resource in ("ac", "dc") else "min_heat_wh"

        active = {r.device_id: r for r in consumers if getattr(r, attr, 0) > 0}
        grants: dict[str, ResourceGrant] = {}
        remaining = available

        # Iteratively curtail devices below minimum until stable
        changed = True
        while changed and active:
            changed = False
            total_requested = sum(getattr(r, attr) for r in active.values())
            if total_requested == 0:
                break
            ratio = min(remaining / total_requested, 1.0)

            to_curtail = []
            for device_id, req in active.items():
                allocation = getattr(req, attr) * ratio
                if allocation < getattr(req, min_attr):
                    to_curtail.append(device_id)

            for device_id in to_curtail:
                grants[device_id] = ResourceGrant.curtailed_grant(device_id)
                del active[device_id]
                changed = True

        # Allocate remaining active consumers
        if active:
            total_requested = sum(getattr(r, attr) for r in active.values())
            ratio = min(remaining / total_requested, 1.0) if total_requested > 0 else 0.0
            for device_id, req in active.items():
                allocation = getattr(req, attr) * ratio
                grants[device_id] = ResourceGrant(
                    device_id=device_id,
                    **{attr: allocation},
                )

        return grants
