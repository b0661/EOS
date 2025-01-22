import os
import time
from http import HTTPStatus

import requests

from akkudoktoreos.server.dash.demo import Demo

if os.name == "nt":
    host = "localhost"
else:
    host = "0.0.0.0"
eosdash_url = f"http://{host}:8504"


def test_eosdash(eosdash_server):
    """Test the EOSdash server."""
    result = requests.get(f"{eosdash_server}/eosdash/health")
    assert result.status_code == HTTPStatus.OK
    assert result.json()["status"] == "alive"


class TestEOSdashServer:
    def test_eosdash_proxied_by_eos(self, server_setup_for_class):
        """Test the EOSdash server proxied by EOS server."""
        # Assure EOSdash is up
        startup = False
        error = ""
        for retries in range(10):
            try:
                result = requests.get(f"{eosdash_url}/eosdash/health")
                if result.status_code == HTTPStatus.OK:
                    startup = True
                    break
                error = f"{result.status_code}, {str(result.content)}"
            except Exception as ex:
                error = str(ex)
            time.sleep(3)
        assert startup, f"Connection to {eosdash_url}/eosdash/health failed: {error}"
        assert result.json()["status"] == "alive"

    def test_eosdash_demo(self, server_setup_for_class):
        """Test EOSdash demo page."""
        server = server_setup_for_class["server"]
        eos_dir = server_setup_for_class["eos_dir"]

        html = Demo(host, 8503)
        assert html is not None
        # Convert to string for easy lookup
        html = str(html)
        assert html.startswith("div")
        assert "function embed_document(root)" in html
