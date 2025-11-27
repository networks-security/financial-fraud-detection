from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1 import Query

# from firebase_admin.db import Query
# from google.cloud.firestore_v1.base_query import Query

cred = credentials.Certificate(r'C:\Things\Important\Work\School\Projects\Network Security\fraud-detection-52ca2-firebase-adminsdk-fbsvc-67f0da74c6.json')
firebase_admin.initialize_app(cred)
db = firestore.client()



private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)
public_key = private_key.public_key()

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
    fraudlisttxt = generateFraudListNumber()
    for txt in range(txtlist):
        txtlist[txt] = convertToBinary(txtlist[txt])
    fraudlisttxt.set({
    'txFraud': encrypt(txtlist[0]),
    'txFraudScenario': encrypt(txtlist[1]),
    'transactionId': encrypt(txtlist[3]),
    'txDatetime': encrypt(txtlist[4]),
    'customerId': encrypt(txtlist[5]),
    'terminalId': encrypt(txtlist[6]),
    'txAmount': encrypt(txtlist[7]),
    'txTimeSeconds': encrypt(txtlist[8]),
    'txTimeDays': encrypt(txtlist[9])})

    return fraudlisttxt

def generateFraudListNumber():
    fraudlisttxt = db.collection('fraudlist')
    query = fraudlisttxt.order_by("name", direction=Query.DESCENDING).limit(1)
    doc = query.get()[0]
    if doc.exists:
        print(f"Document ID: {doc.id}")
    else:
        print("Document does not exist.")

    return fraudlisttxt

generateFraudListNumber()

# fraudlisttxt2.set({
#     'txFraud': encrypt(b'56'),
#     'txFraudScenario': encrypt(b'120'),
#     'transactionId': encrypt(b'12314112'),
#     'txDatetime': encrypt(b'27/1/2025'),
#     'customerId': encrypt(b'233281'),
#     'terminalId': encrypt(b'12334'),
#     'txAmount': encrypt(b'4000.00'),
#     'txTimeSeconds': encrypt(b'2348'),
#     'txTimeDays': encrypt(b'23'),
# })

users_ref = db.collection('fraudlist')
docs = users_ref.stream()

for doc in docs:
    print(f'{doc.id} => {doc.to_dict()}')

def decrypt(encrypted_data):
    decrypted_data = private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

