"""Provides feed in tariff data."""

from typing import Optional

from loguru import logger
from pydantic import Field

from akkudoktoreos.config.configabc import SettingsBaseModel
from akkudoktoreos.prediction.feedintariffabc import FeedInTariffProvider
from akkudoktoreos.utils.datetimeutil import to_datetime


class FeedInTariffFixedCommonSettings(SettingsBaseModel):
    """Common settings for elecprice fixed price."""

    feed_in_tariff_amt_kwh: float = Field(
        default=0.0,
        ge=0,
        json_schema_extra={
            "description": "Electricity feed in tariff [Amt./kWH].",
            "examples": [0.078],
        },
    )


class FeedInTariffFixed(FeedInTariffProvider):
    """Fixed price feed in tariff data.

    FeedInTariffFixed is a singleton-based class that retrieves elecprice data.
    """

    @classmethod
    def provider_id(cls) -> str:
        """Return the unique identifier for the FeedInTariffFixed provider."""
        return "FeedInTariffFixed"

    def _update_data(self, force_update: Optional[bool] = False) -> None:
        error_msg = "Feed in tariff not provided"
        try:
            feed_in_tariff_amt_kwh = (
                self.config.feedintariff.feedintarifffixed.feed_in_tariff_amt_kwh
            )
        except:
            logger.exception(error_msg)
            raise ValueError(error_msg)
        if feed_in_tariff_amt_kwh is None:
            logger.error(error_msg)
            raise ValueError(error_msg)
        feed_in_tariff_amt_wh = feed_in_tariff_amt_kwh / 1000
        self.update_value(to_datetime(), "feed_in_tariff_amt_wh", feed_in_tariff_amt_wh)
