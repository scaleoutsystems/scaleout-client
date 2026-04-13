from config import settings
from scaleout import Scaleout

client = Scaleout(
    host=settings["DISCOVER_HOST"],
    port=settings["DISCOVER_PORT"],
    secure=settings["SECURE"],
    verify=settings["VERIFY"],
    token=settings["ADMIN_TOKEN"],
)

result = client.set_active_model("seed.npz")
print(result["message"])
