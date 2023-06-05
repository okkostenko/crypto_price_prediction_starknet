import json
import subprocess
import os
# import boto3
import re

TEST_CONTRACT_ADDRESS = "0x0000000" # TODO: change this to the correct address
ABI_FILE = "contract_starter/abi.json"
SCALE_FACTOR = 1e8
LOAD_DIR = "model_training/models/mlp/class/weights"
STRAKNET_NETWORK = "alpha-goerli"
STARKNET_WALLET = "starkware.starknet.wallets.open_zeppelin.OpenZeppelinAccount"

def read_most_recent_model_data_file():
    ...

def read_most_recent_input_data_file():
    ...

def get_cairo_command(contract_address:str, abi_filepath:str, fn_name:str, args:str=None, invocative_type="call"):

    output = f"/hoe/aweso/cairo-venn/bin/starknet {invocative_type} --address {contract_address} --abi {abi_filepath} --function {fn_name}"

    if args is not None:
        ret += f" --inputs {args}"
    
    return ret

def get_flattened_array(array):
    return " ".join([str(x) for x in array])

def test_trading_bot_mlp(x:dict, weights_json:dict, price_ratio:float, remaining_usdt:float, remaining_weth:float, scale_factor:int, invoke=True):
    input_str = f"{remaining_weth} {remaining_usdt} {price_ratio} {x.get('n_cols')} {x.get('n_rows')} {x.get('n_cols') * x.get('n_rows')}  {get_flattened_array(x.get('data'))} "

    for layer in weights_json.get("layers"):
        input_str += f"{layer.get('n_cols')} {layer.get('n_rows')} {layer.get('n_cols') * layer.get('n_rows')} {get_flattened_array(layer.get('data'))} {layer.get('n_cols')} {get_flattened_array(layer.get('bias'))} "

    input_str += str(int(scale_factor))

    cairo_command = get_cairo_command(TEST_CONTRACT_ADDRESS, 
                                      ABI_FILE, 
                                      "calculate_strategy", 
                                      input_str,
                                      invocation_type="invoke" if invoke else "call")
    
    print(f"Executing Cairo command: {cairo_command}")

    output = subprocess.run(cairo_command.split(" "), 
                            capture_output=True, 
                            env={"STARTKNET_NETWORK": STRAKNET_NETWORK, "STARKNET_WALLET": STARKNET_WALLET})
    print(output)
    return output.stdout


def handler(event, context):
    x = read_most_recent_input_data_file()
    model_data = read_most_recent_model_data_file()
    # TODO: read remaining coins and priceratio
    output = test_trading_bot_mlp(x, model_data) # TODO: add the correct arguments
    transaction_hash = re.search(r"(Transaction hash: )([a-zA-Z0-9]+)", output.decode('utf-8')).group(2)
    print(transaction_hash)

    with open("transaction_hash.json", "w") as f: # TODO: change this to the correct file
        f.write(json.dumps(transaction_hash))
    
    return 0

if __name__ == "__main__":
    handler((), ())
