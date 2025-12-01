import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyBo1tzoIMIm8uEzViuyrh3QsO69moNLJvg",
  authDomain: "fraud-detection-52ca2.firebaseapp.com",
  databaseURL: "https://fraud-detection-52ca2-default-rtdb.firebaseio.com",
  projectId: "fraud-detection-52ca2",
  storageBucket: "fraud-detection-52ca2.firebasestorage.app",
  messagingSenderId: "747355499474",
  appId: "1:747355499474:web:302ec2db9ffba7e2a3360b",
  measurementId: "G-TF2S1YT6YR",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// export const analytics = getAnalytics(app);
export const auth = getAuth(app);
