import express, { Request, Response } from 'express';
import { VoiceRecording } from './voiceRecording.model';
import { authenticate } from '../auth/auth.middleware';
import multer from 'multer';
import { uploadImage, uploadAudio } from '../../utils/cloudinary';
import { exec } from 'child_process';
import path from 'path';
import fs from 'fs';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

// Task 1: Get all recorded voices for the home page with a configurable limit and optional title filter
router.get('/recordings', authenticate, async (req: Request, res: Response): Promise<void> => {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({ success: false, message: 'Unauthorized' });
        return;
      }
  
      const limitParam = req.query.limit as string;
      const titleParam = req.query.title as string;
  
      let limit = 5;
      if (limitParam) {
        const parsedLimit = parseInt(limitParam);
        if (!isNaN(parsedLimit) && parsedLimit > 0) {
          limit = parsedLimit;
        } else {
          console.log('Step 4: Invalid limit provided, using default value of 5');
          res.status(400).json({ success: false, message: 'Limit must be a positive integer' });
          return;
        }
      }
  
      // Step 5: Build query with optional title filter
      const query: any = { user: userId };
      if (titleParam) {
        query.title = { $regex: titleParam, $options: 'i' }; // case-insensitive partial match
      }
  
      console.log('Step 6: Running query with filters:', query);
  
      const recordings = await VoiceRecording.find(query).limit(limit);
      res.status(200).json({ success: true, recordings });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  });
  

// Task 1: Add a new voice recording (new or uploaded)
router.post(
  '/add-voice',
  authenticate,
  upload.fields([{ name: 'audio', maxCount: 1 }, { name: 'image', maxCount: 1 }]),
  async (req: Request, res: Response): Promise<void> => {
    try {
      console.log('Step 1: Starting /add-voice route');
      const userId = req.user?.id;
      console.log('Step 2: Extracted userId:', userId);
      if (!userId) {
        console.log('Step 3: No userId found, returning 401');
        res.status(401).json({ success: false, message: 'Unauthorized' });
        return;
      }

      const { title, relation, ...personalizationData } = req.body;
      console.log('Step 4: Extracted title, relation, and personalizationData:', { title, relation, personalizationData });
      let parsedPersonalizationData = { ...personalizationData };
      if (personalizationData.customQuestions && typeof personalizationData.customQuestions === 'string') {
        try {
          parsedPersonalizationData.customQuestions = JSON.parse(personalizationData.customQuestions);
        } catch (e) {
          console.log('Step 4.5: Failed to parse customQuestions:', e);
          parsedPersonalizationData.customQuestions = [];
        }
      }

      const files = req.files as { [fieldname: string]: Express.Multer.File[] };
      console.log('Step 5: Extracted files:', files);

      if (!title || !relation) {
        console.log('Step 6: Missing title or relation, returning 400');
        res.status(400).json({ success: false, message: 'Title and relation are required' });
        return;
      }

      if (!files.audio || files.audio.length === 0) {
        console.log('Step 7: No audio file provided, returning 400');
        res.status(400).json({ success: false, message: 'Audio file is required' });
        return;
      }

      console.log('Step 8: Uploading audio to Cloudinary:', files.audio[0].path);
      const audioResult = await uploadAudio(files.audio[0].path);
      const audioUrl = audioResult.secure_url;
      console.log('Step 9: Audio uploaded, URL:', audioUrl);

      let imageUrl = '';
      if (files.image && files.image.length > 0) {
        console.log('Step 10: Uploading image to Cloudinary:', files.image[0].path);
        const imageResult = await uploadImage(files.image[0].path);
        imageUrl = imageResult.secure_url;
        console.log('Step 11: Image uploaded, URL:', imageUrl);
      }

      const pythonScriptPath = path.join(__dirname, '../../../../python/audio_cloning.py');
      const tempAudioPath = files.audio[0].path.replace(/\\/g, '/');
      const cloneName = `${title}-${userId}`.replace(/ /g, '_');
      console.log('Step 12: Preparing to call Python script:', { pythonScriptPath, tempAudioPath, cloneName });

      const clonedVoiceId = await new Promise<string>((resolve, reject) => {
        const command = `python -u "${pythonScriptPath}" "${tempAudioPath}" "${cloneName}"`;
        console.log('Step 12.5: Executing command:', command);
        exec(command, { shell: 'cmd.exe' }, (error, stdout, stderr) => {
          console.log('Step 13: Python script executed, stdout:', stdout);
          console.log('Step 14: Python script executed, stderr:', stderr);
          if (error) {
            console.log('Step 15: Python script execution error:', error);
            reject(new Error(`Voice cloning failed: ${stderr || error.message}`));
            return;
          }
          const voiceIdMatch = stdout.match(/Cloned voice ID: (.+)/);
          const voiceId = voiceIdMatch ? voiceIdMatch[1].trim() : null;
          console.log('Step 16: Extracted voiceId:', voiceId);
          if (!voiceId) {
            console.log('Step 17: No voiceId extracted, rejecting promise');
            reject(new Error('Failed to extract cloned voice ID'));
            return;
          }
          resolve(voiceId);
        });
      });

      console.log('Step 18: Cloned voice ID obtained:', clonedVoiceId);

      console.log('Step 19: Cleaning up temp audio file:', tempAudioPath);
      fs.unlinkSync(tempAudioPath);
      if (files.image) {
        console.log('Step 20: Cleaning up temp image file:', files.image[0].path);
        fs.unlinkSync(files.image[0].path);
      }

      console.log('Step 21: Saving voice recording to database');
      const voiceRecording = new VoiceRecording({
        user: userId,
        title,
        relation,
        imageUrl,
        audioUrl,
        clonedVoiceId,
        personalizationData: parsedPersonalizationData,
      });

      await voiceRecording.save();
      console.log('Step 22: Voice recording saved:', voiceRecording);

      res.status(201).json({ success: true, voiceRecording });
    } catch (error: any) {
      console.error('Step 23: Error in /add-voice:', error.message);
      res.status(500).json({ success: false, message: error.message });
    }
  }
);

// Task 2: Update personalization data and profile picture for a voice recording
router.put(
    '/:recordingId/personalization',
    authenticate,
    upload.single('image'), // Add multer to handle image upload
    async (req: Request, res: Response): Promise<void> => {
      try {
        const userId = req.user?.id;
        const { recordingId } = req.params;
        const { personalizationData } = req.body;
        const file = req.file; // Image file from multer
  
        if (!userId) {
          res.status(401).json({ success: false, message: 'Unauthorized' });
          return;
        }
  
        const voiceRecording = await VoiceRecording.findOne({ _id: recordingId, user: userId });
        if (!voiceRecording) {
          res.status(404).json({ success: false, message: 'Voice recording not found' });
          return;
        }
  
        const updateFields: { [key: string]: any } = {};
  
        // Handle personalization data updates
        if (personalizationData) {
          let parsedPersonalizationData = personalizationData;
          if (typeof personalizationData === 'string') {
            try {
              parsedPersonalizationData = JSON.parse(personalizationData);
            } catch (e) {
              console.log('Failed to parse personalizationData:', e);
              parsedPersonalizationData = {};
            }
          }
          for (const key in parsedPersonalizationData) {
            if (parsedPersonalizationData.hasOwnProperty(key)) {
              updateFields[`personalizationData.${key}`] = parsedPersonalizationData[key];
            }
          }
        }
  
        // Handle image upload
        if (file) {
          console.log('Uploading new profile image to Cloudinary:', file.path);
          const imageResult = await uploadImage(file.path);
          updateFields.imageUrl = imageResult.secure_url;
          console.log('New image URL:', imageResult.secure_url);
          fs.unlinkSync(file.path); // Clean up temporary file
        }
  
        if (Object.keys(updateFields).length === 0) {
          res.status(400).json({ success: false, message: 'No updates provided' });
          return;
        }
  
        const updatedRecording = await VoiceRecording.findOneAndUpdate(
          { _id: recordingId, user: userId },
          { $set: updateFields },
          { new: true, runValidators: true }
        );
  
        if (!updatedRecording) {
          res.status(404).json({ success: false, message: 'Voice recording not found' });
          return;
        }
  
        res.status(200).json({ success: true, voiceRecording: updatedRecording });
      } catch (error: any) {
        console.error('Error in update personalization:', error.message);
        if (req.file) {
          fs.unlinkSync(req.file.path); // Clean up on error
        }
        res.status(500).json({ success: false, message: error.message });
      }
    }
  );

// Task 3: Talk to AI with selected voice
router.post(
  '/talk-to-ai',
  authenticate,
  async (req: Request, res: Response): Promise<void> => {
    try {
      const userId = req.user?.id;
      console.log('Step 1: Extracted userId:', userId);
      if (!userId) {
        res.status(401).json({ success: false, message: 'Unauthorized' });
        return;
      }

      const { recordingId, userInput } = req.body;
      console.log('Step 2: Received recordingId and userInput:', { recordingId, userInput });
      if (!recordingId || !userInput) {
        res.status(400).json({ success: false, message: 'Recording ID and user input are required' });
        return;
      }

      const voiceRecording = await VoiceRecording.findOne({ _id: recordingId, user: userId });
      console.log('Step 3: Fetched voiceRecording:', voiceRecording);
      if (!voiceRecording || !voiceRecording.clonedVoiceId) {
        res.status(404).json({ success: false, message: 'Voice recording or cloned voice not found' });
        return;
      }

      const tempFilePath = path.join(__dirname, '../../../../uploads/temp_user_data.json');
      fs.writeFileSync(tempFilePath, JSON.stringify(voiceRecording.personalizationData));
      console.log('Step 4: Wrote userData to temporary file:', tempFilePath);

      const pythonScriptPath = path.join(__dirname, '../../../../python/generate_ai_response.py');
      const outputAudioPath = path.join(__dirname, '../../../../uploads/generated_audio.wav');
      const command = `python -u "${pythonScriptPath}" "${userInput}" ${voiceRecording.clonedVoiceId} "${tempFilePath}" "${outputAudioPath}"`;
      console.log('Step 5: Executing command:', command);

      const generatedAudioPath = await new Promise<string>((resolve, reject) => {
        exec(
          command,
          { shell: 'cmd.exe' },
          (error, stdout, stderr) => {
            console.log('Step 6: Python script executed, stdout:', stdout);
            console.log('Step 7: Python script executed, stderr:', stderr);
            if (error) {
              console.log('Step 8: Python script execution error:', error);
              reject(new Error(`AI response generation failed: ${stderr || error.message}`));
              return;
            }
            // Improved parsing logic to handle multiple lines
            console.log('Step 9: Full stdout for debugging:', stdout);
            const audioPathMatch = stdout.match(/Generated audio saved at: (.+)/);
            const audioPath = audioPathMatch ? audioPathMatch[1].trim() : null;
            console.log('Step 10: Extracted audioPath:', audioPath);
            if (!audioPath) {
              console.log('Step 11: No audioPath extracted, rejecting promise. Full stdout:', stdout);
              reject(new Error('Failed to extract generated audio path'));
              return;
            }
            resolve(audioPath);
          }
        );
      });

      fs.unlinkSync(tempFilePath);
      console.log('Step 12: Cleaned up temporary userData file');

      console.log('Step 13: Uploading generated audio to Cloudinary:', generatedAudioPath);
      const audioResult = await uploadAudio(generatedAudioPath);
      console.log('Step 14: Audio uploaded, URL:', audioResult.secure_url);
      fs.unlinkSync(generatedAudioPath);
      console.log('Step 15: Cleaned up temporary audio file');

      res.status(200).json({ success: true, audioUrl: audioResult.secure_url });
    } catch (error: any) {
      console.error('Step 16: Error in /talk-to-ai:', error.message);
      res.status(500).json({ success: false, message: error.message });
    }
  }
);


// Task 1: Delete a voice recording by ID
router.delete(
    '/:recordingId',
    authenticate,
    async (req: Request, res: Response): Promise<void> => {
      try {
        const userId = req.user?.id;
        const { recordingId } = req.params;
  
        if (!userId) {
          res.status(401).json({ success: false, message: 'Unauthorized' });
          return;
        }
  
        const voiceRecording = await VoiceRecording.findOneAndDelete({
          _id: recordingId,
          user: userId,
        });
  
        if (!voiceRecording) {
          res.status(404).json({ success: false, message: 'Voice recording not found or you do not have permission to delete it' });
          return;
        }
  
        res.status(200).json({ success: true, message: 'Voice recording deleted successfully' });
      } catch (error: any) {
        console.error('Error in delete recording:', error.message);
        res.status(500).json({ success: false, message: error.message });
      }
    }
  );

export default router;