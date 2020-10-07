from azureml.core import Workspace
from azureml.core.authentication import AuthenticationException, AzureCliAuthentication


def get_auth():
    try:
        auth = AzureCliAuthentication()
        auth.get_authentication_header()
    except AuthenticationException:
        print("Auth Error")
    return auth

print("Enter setup.py")

print("Trying to create Workspace with CLI Authentication")
print("Workspace access from config using CLI Auth")
ws = Workspace.from_config(auth=get_auth())
print("Workspace access from config successful")

print("Workspace Details")
print(ws.get_details())

#print("Auth End")
print("Success  setup.py")