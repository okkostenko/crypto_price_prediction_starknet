from starknet_py import Provider as Sn_Provider
from ethers import ethers
from l1_contracts.typechain_types.factories.contracts.RockafellerBotL1.sol.RockafellerBotL1__factory import RockafellerBotL1__factory
import json

from l1_contarcts.keys_config import MAINNET_PRIVATE_KEY

from firebase_amin import initialize_app, credential, firestore
from firebase_scripts.types.firebase import TradesDocument
from firebase.firestore import Timestamp

import time
import asyncio

firebaseConfig = {
    "apiKey": "",
    "authDomain": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "credential": credential.Certificate("") # TODO: change this to the correct config
}

L1_CONTRACT_ADDRESS = "0x00000" # TODO: change this to the correct address

async def handler():
    sn_provider = Sn_Provider({
        "network": "mainnet-alpha"  # or "goerli-alpha"
    })

    with open("./dev/transHashData/current_trans_hash.json", "r") as f: # TODO: change this to the correct file
        trans = json.load(f)

    while True:
        print("Checking if trans is accepted on L1")
        t_receipt = await sn_provider.getTransactionReceipt(trans)
        if (t_receipt.status == "ACCEPTED_ON_L2" or t_receipt.status == "ACCEPTED_ON_L1") and len(t_receipt.l2_to_l1_messages) == 0:
            app = initialize_app(firebaseConfig)
            db = firestore.client(app)
            current_time = int(time.time())
            trade_document = TradesDocument(
                action_type="HOLD",
                amount="0",
                timestamp=Timestamp(current_time, 0)
            ) # TODO: check if everything is correct here
            await db.collection("trades").document(str(current_time)).set(trade_document)
            return
        if t_receipt.status != "ACCEPTED_ON_L1":
            await asyncio.sleep(120)
        else:
            break

    message = t_receipt.l2_to_l1_messages[0]
    print(message)
    trade_instruction = message.payload[0]
    trade_amount = message.payload[1]
    provider = ethers.providers.AlchemyProvider("mainnet", "-HC6HC2R9crhQe3tcu4h4Qq2nYfCRc9t") # TODO: change this to the test net

    owner = ethers.Wallet(MAINNET_PRIVATE_KEY, provider)

    RfB = RockafellerBotL1__factory.connect(L1_CONTRACT_ADDRESS, owner)

    print("Starting Trade!")

    add_funds_trans = await RfB.receiveInstruction(trade_instruction, trade_amount, {"gasLimit": 1000000}) # TODO: change the gas limit
    await add_funds_trans.wait()
    print("Trade Complete!")

asyncio.run(handler())