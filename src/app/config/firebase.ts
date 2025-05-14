import admin from 'firebase-admin';

// Load Firebase service account from environment variable
const serviceAccountRaw = process.env.FIREBASE_SERVICE_ACCOUNT;

let serviceAccount: admin.ServiceAccount;
if (serviceAccountRaw) {
  try {
    // Step 1: Log the raw value for debugging
    console.log('Raw FIREBASE_SERVICE_ACCOUNT before processing:', JSON.stringify(serviceAccountRaw));

    // Step 2: Parse the initial JSON string
    let parsedServiceAccount = JSON.parse(serviceAccountRaw);

    // Step 3: Handle the private_key to restore proper PEM format
    if (parsedServiceAccount.private_key) {
      // Replace \\n with actual newlines and format as multi-line PEM
      parsedServiceAccount.private_key = parsedServiceAccount.private_key
        .replace(/\\n/g, '\n') // Convert escaped newlines to actual newlines
        .trim(); // Remove trailing spaces
    }

    // Step 4: Log the adjusted object for debugging
    console.log('Adjusted FIREBASE_SERVICE_ACCOUNT before final parsing:', JSON.stringify(parsedServiceAccount, null, 2));

    // Step 5: Assign to serviceAccount
    serviceAccount = parsedServiceAccount as admin.ServiceAccount;

    // Step 6: Log the final object for debugging
    console.log('Final FIREBASE_SERVICE_ACCOUNT:', serviceAccount);
  } catch (error) {
    console.error('Error processing FIREBASE_SERVICE_ACCOUNT:', {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });
    throw new Error(`Failed to process FIREBASE_SERVICE_ACCOUNT: ${error instanceof Error ? error.message : String(error)}`);
  }
} else {
  console.error('FIREBASE_SERVICE_ACCOUNT environment variable is not set');
  throw new Error('FIREBASE_SERVICE_ACCOUNT environment variable is not set');
}

if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
}

export const messaging = admin.messaging();