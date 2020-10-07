import argparse 
from azureml.core import Workspace


# Initialize parser 
parser = argparse.ArgumentParser() 
parser.add_argument("-sid", "--subscription_id", help = "Subscription ID")
parser.add_argument("-rg", "--resource_group", help = "Resource Group")
parser.add_argument("-wn", "--workspace_name", help = "Workspace  Name")
args = parser.parse_args()


ws = Workspace(subscription_id=args.subscription_id, 
            resource_group=args.resource_group, 
            workspace_name= args.workspace_name)


print("Workspace Details")
print(ws.get_details())