
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1 import Query
import sys
import json

# Getting JSON array from script call arg
if len(sys.argv) > 1:
    fileName = sys.argv[1]
    try:
        with open(fileName, 'r') as file:
            array = json.load(file)
        txt = array
        print(f"Received array in Python from file: {txt}")
    except FileNotFoundError:
        print("Error: 'data.json' not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file.")

# Creating a private key (only for the first time)
def createPrivateKey():
    private_key = serialization.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.BestAvailableEncryption(b"NetworkSecurity")
    )
    with open("src/database/private_key_encrypted.pem", "wb") as f:
        f.write(pem)

# Loading a private key
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

# Getting public key from private key (we need to figure out how to make it more secure later)
def getPublicKeyFromPrivateKey():
    public_key = privateKeyAcquired.public_key()
    return public_key

# getting firebase client
def getClient():
    cred = credentials.Certificate('src/database/fraud-detection-52ca2-firebase-adminsdk-fbsvc-b77f6e51e6.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db

# encryption method (if I have some time after this I will make it 3 layers of encryption (encrypt)(decrypt)(encrypt))
def encrypt(plaintext):
    encrypted_data = public_key.encrypt(
        str(plaintext).encode('ascii'),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_data

# binary conversion method
def convertToBinary(txt):
    bin = ''.join(format(ord(char), '08b') for char in txt)
    return bin

# posting the list to firestore
def postList(txtlist, db):

    # for txtNum in range(len(txtlist)):
    #     txtlist[txtNum] = convertToBinary(txtlist[txtNum])
    num = generateFraudListNumber(db)
    fraudlisttxt = db.collection('fraudlist').document(str(num))

    fraudlisttxt.set({
    'txFraud': encrypt(txtlist["txFraud"]),
    'txFraudScenario': encrypt(txtlist["txFraudScenario"]),
    'transactionId': encrypt(txtlist["transactionId"]),
    'txDatetime': encrypt(txtlist["txDatetime"]),
    'customerId': encrypt(txtlist["customerId"]),
    'terminalId': encrypt(txtlist["terminalId"]),
    'txAmount': encrypt(txtlist["txAmount"]),
    'txTimeSeconds': encrypt(txtlist["txTimeSeconds"]),
    'txTimeDays': encrypt(txtlist["txTimeDays"]),
    'userId': encrypt(txtlist["userId"]),
    'txID': num,
    })

# automatically increment document ID based on highest present docuement ID
def generateFraudListNumber(db):
    fraudlisttxt = db.collection('fraudlist')
    query = fraudlisttxt.order_by("txID", direction=Query.DESCENDING).limit(2)
    docs = query.get()
    try:
        doc = docs[0]
    except IndexError:
        doc = docs[0]
    if doc.exists:
        print(f"Document ID: {doc.id}")
        newNum = int(doc.id) + 1
        return newNum
    else:
        print("Document does not exist.")


# run
privateKeyAcquired = getPrivateKey()
public_key = getPublicKeyFromPrivateKey()
db = getClient()
postList(txt, db)



