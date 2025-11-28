from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1 import Query
import sys
import json

# fraudlisttxt = db.collection('fraudlist').document('3')
txt = [b'123', b'3124', b'12314', b'29/3/2024', b'1231', b'44333', b'90.00', b'2313', b'89']

if len(sys.argv) > 1:
    txt = sys.argv[1]

private_key_path = "private_key_encrypted.pem"
with open(private_key_path, "rb") as key_file:
    private_key_data = key_file.read()
password = b"NetworkSecurity"
privateKeyAcquired = serialization.load_pem_private_key(
    private_key_data,
    password=password,
    backend=default_backend()
)
public_key = privateKeyAcquired.public_key()

cred = credentials.Certificate(r'fraud-detection-52ca2-firebase-adminsdk-fbsvc-b77f6e51e6.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def encrypt(plaintext):
    encrypted_data = public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_data

def convertToBinary(txt):
    bin = ''.join(format(ord(char), '08b') for char in txt)
    return bin


def encryptList(txtlist):
    # for txtNum in range(len(txtlist)):
    #     txtlist[txtNum] = convertToBinary(txtlist[txtNum])
    num = generateFraudListNumber()
    fraudlisttxt = db.collection('fraudlist').document(str(num))

    fraudlisttxt.set({
    'txFraud': encrypt(txtlist[0]),
    'txFraudScenario': encrypt(txtlist[1]),
    'transactionId': encrypt(txtlist[2]),
    'txDatetime': encrypt(txtlist[3]),
    'customerId': encrypt(txtlist[4]),
    'terminalId': encrypt(txtlist[5]),
    'txAmount': encrypt(txtlist[6]),
    'txTimeSeconds': encrypt(txtlist[7]),
    'txTimeDays': encrypt(txtlist[8]),
    'txID': num,
    })

def generateFraudListNumber():
    fraudlisttxt = db.collection('fraudlist')
    query = fraudlisttxt.order_by("txID", direction=Query.DESCENDING).limit(2)
    docs = query.get()
    doc = docs[1]
    if doc.exists:
        print(f"Document ID: {doc.id}")
        newNum = int(doc.id) + 1
        return newNum
    else:
        print("Document does not exist.")

encryptList(txt)


