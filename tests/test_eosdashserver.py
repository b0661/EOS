import os
import time
from http import HTTPStatus

import requests


def test_eosdash(eosdash_server):
    """Test the EOSdash server."""
    result = requests.get(f"{eosdash_server}/eosdash/health")
    assert result.status_code == HTTPStatus.OK
    assert result.json()["status"] == "alive"


def test_eosdash_proxied_by_eos(server):
    """Test the EOSdash server proxied by EOS server."""
    if os.name == "nt":
        host = "localhost"
    else:
        host = "0.0.0.0"
    eosdash_server = f"http://{host}:8504"

    # Assure EOSdash is up
    startup = False
    error = ""
    for retries in range(10):
        try:
            result = requests.get(f"{eosdash_server}/eosdash/health")
            if result.status_code == HTTPStatus.OK:
                startup = True
                break
            error = f"{result.status_code}, {str(result.content)}"
        except Exception as ex:
            error = str(ex)
        time.sleep(3)
    assert startup, f"Connection to {eosdash_server}/eosdash/health failed: {error}"
    assert result.json()["status"] == "alive"
