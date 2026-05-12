from server_functions import ServerFunctions

from scaleout import Scaleout

# Fetch your host address from the studio UI and add it below.
client = Scaleout(host="", secure=True, verify=True)

print(client.start_session(server_functions=ServerFunctions))
