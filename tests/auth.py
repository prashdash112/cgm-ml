import argparse
from azureml.core import Workspace
#from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.authentication import AuthenticationException, AzureCliAuthentication

def get_auth():
    try:
        auth = AzureCliAuthentication()
        auth.get_authentication_header()
    except AuthenticationException:
        print("Auth Error")
    return auth



# Initialize parser 
parser = argparse.ArgumentParser() 
#parser.add_argument("-tid", "--tenant_id", help = "Tenant Id")
#parser.add_argument("-spid", "--service_principal_id", help = "Service Principal Id")
#parser.add_argument("-sppwd", "--service_principal_password", help = "Service Principal Password")

parser.add_argument("-sid", "--subscription_id", help = "Subscription ID")
parser.add_argument("-rg", "--resource_group", help = "Resource Group")
parser.add_argument("-wn", "--workspace_name", help = "Workspace  Name")


args = parser.parse_args()

'''
print("Trying to create Workspace with CLI Authentication")

svc_pr = ServicePrincipalAuthentication(
    tenant_id = args.tenant_id,
    service_principal_id = args.service_principal_id,
    service_principal_password = args.service_principal_password)

print("Get auth Header")

print(svc_pr.get_authentication_header())

print("Login Successful")


ws = Workspace(subscription_id = args.subscription_id, 
    resource_group = args.resource_group, 
    workspace_name = args.workspace_name,
    auth=svc_pr)


print("Found workspace {} at location {}".format(ws.name, ws.location))

print("Workspace Details")
print(ws.get_details())

print("Auth End")


print("Save config files")
ws.write_config()


print("Workspace access from config with service principal")
ws = Workspace.from_config(auth=svc_pr)

print("Workspace access from config successful")

print("Workspace Details")
print(ws.get_details())

print("Auth One End")

'''

print("CLI Auth started")
auth =  get_auth()
print("CLI Auth success")

print("Workspace setup started")

ws = Workspace(subscription_id = args.subscription_id, 
    resource_group = args.resource_group, 
    workspace_name = args.workspace_name,
    auth=get_auth())

print("Workspace Details")
print(ws.get_details())

print("Workspace setup success")


ws.write_config()
print("Saved config file")

print("Trying to create Workspace with CLI Authentication")
print("Workspace access from config using CLI Auth")
ws = Workspace.from_config(auth=get_auth())
print("Workspace access from config successful")

print("Workspace Details")
print(ws.get_details())

print("Auth End")
