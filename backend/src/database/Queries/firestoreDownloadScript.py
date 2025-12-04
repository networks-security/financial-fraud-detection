from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1 import Query
import sys
import json

def getPrivateKey():
    private_key_path = "src/database/private_key_encrypted.pem"
    with open(private_key_path, "rb") as key_file:
        private_key_data = key_file.read()
    password = b"NetworkSecurity"
    privateKeyAcquired = serialization.load_pem_private_key(
        private_key_data,
        password=password,
        backend=default_backend()
    )
    return privateKeyAcquired

def getClient():
    cred = credentials.Certificate('src/database/fraud-detection-52ca2-firebase-adminsdk-fbsvc-b77f6e51e6.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db

def decrypt(ciphertext, privateKey):
    decryptedText = privateKey.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    return decryptedText.decode('ascii')

def fetchData(db, privateKey):
    collection_name = "fraudlist"
    docs = db.collection(collection_name).stream()
    print(docs)
    data = []
    for doc in docs:
        print(doc)
        docData = doc.to_dict()
        # print(docData)
        # print("\n\n\n\n\n\n\n")
        docDataDecrypted = {
            'txFraud': decrypt(docData["txFraud"], privateKey),
            'txFraudScenario': decrypt(docData["txFraudScenario"], privateKey),
            'transactionId': decrypt(docData["transactionId"], privateKey),
            'txDatetime': decrypt(docData["txDatetime"], privateKey),
            'customerId': decrypt(docData["customerId"], privateKey),
            'terminalId': decrypt(docData["terminalId"], privateKey),
            'txAmount': decrypt(docData["txAmount"], privateKey),
            'txTimeSeconds': decrypt(docData["txTimeSeconds"], privateKey),
            'txTimeDays': decrypt(docData["txTimeDays"], privateKey),
            'userID': decrypt(docData["userID"], privateKey)
            }
        docDataDecrypted['id'] = doc.id 
        data.append(docDataDecrypted)
        print("completed")
    return data

def jsonifyData(data):
    outputFileName = "fraudlist.json"

    with open(outputFileName, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfuly exported")

# run
privateKeyAcquired = getPrivateKey()
db = getClient()
data = fetchData(db, privateKeyAcquired)
jsonifyData(data)