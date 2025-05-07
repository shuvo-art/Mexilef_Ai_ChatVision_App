import admin from 'firebase-admin';
import { readFileSync } from 'fs';

// Load Firebase service account from environment variable
const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT || '{}');

if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount as admin.ServiceAccount),
  });
}

export const messaging = admin.messaging();





/* import admin from 'firebase-admin';
import serviceAccount from './somos-ai-voice-cloning-firebase-adminsdk-fbsvc-0793bb60e0.json'; 

// Initialize Firebase Admin SDK
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount as admin.ServiceAccount),
  });
}

export const messaging = admin.messaging(); */