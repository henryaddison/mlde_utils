import logging

logger = logging.getLogger(__name__)


class SelectGCMDomain:

    DOMAIN_CENTRES = {
        "birmingham": {
            9: dict(
                longitude=slice(209, 218),
                latitude=slice(252, 261),
            ),
        },
    }

    def __init__(self, subdomain, size) -> None:
        self.subdomain = subdomain
        self.size = size

    def run(self, ds):
        logger.info(f"Selecting GCM subdomain {self.subdomain}")

        return ds.isel(**self.DOMAIN_CENTRES[self.subdomain][self.size])
