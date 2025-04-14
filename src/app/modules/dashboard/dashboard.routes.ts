import express from 'express';
import { getAllUsers, getUserDetails, deleteConversation, suspendUserSubscription, uploadAHeroImage, getAllHeroImages, deleteHeroImageController } from './dashboard.controller';
import { authenticate, requireRole } from '../auth/auth.middleware';
import multer from 'multer';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

// Route to fetch all users with subscription details (Task 1)
router.get('/users', requireRole('admin'), getAllUsers);

// Route to fetch a specific user's details and chat history (unchanged)
router.get('/user/:userId', requireRole('admin'), getUserDetails);

// Route to delete a specific conversation (unchanged)
router.delete('/conversation/:conversationId', requireRole('admin'), deleteConversation);

// Route to suspend a user's subscription (Task 1)
router.put('/suspend-subscription/:userId', requireRole('admin'), suspendUserSubscription);

// Route to upload hero section image (Task 3)
router.post('/upload-hero-image', requireRole('admin'), upload.single('image'), uploadAHeroImage);

// New route to retrieve all hero section images
router.get('/hero-images', authenticate, getAllHeroImages);

// New route to delete a hero image
router.delete('/hero-image/:publicId', requireRole('admin'), deleteHeroImageController);

export default router;