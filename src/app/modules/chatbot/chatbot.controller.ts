import { RequestHandler } from 'express';
import { ChatHistory } from './chatHistory.model';
import { exec } from 'child_process';
import path from 'path';
import fs from 'fs';
import { promisify } from 'util';
import { uploadImage } from '../../utils/cloudinary';

const execPromise = promisify(exec);

interface UserInputData {
  text_input?: string;
  pdf_path?: string;
  image_path?: string;
}

interface ResponseData {
  response: string;
}

async function processUserInput({ text_input, pdf_path, image_path }: UserInputData): Promise<ResponseData> {
  const pythonScriptPath = '/app/maxim/main.py'; // Absolute path inside container
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  let command = `${pythonCmd} -u "${pythonScriptPath}"`;
  if (pdf_path) command += ` --upload "${pdf_path}"`;
  if (image_path) command += ` --image "${image_path}"`;
  if (text_input) command += ` "${text_input}"`;
  console.log('Executing Python command:', command);

  const execOptions = {
    cwd: path.dirname(pythonScriptPath), // Set working directory to maxim directory
    env: {
      ...process.env, // Pass all existing environment variables
      OPENAI_API_KEY: process.env.OPENAI_API_KEY,
      VISION_API_KEY: process.env.VISION_API_KEY,
    },
  };

  try {
    const { stdout, stderr } = await execPromise(command, execOptions);
    console.log('Python script stdout:', stdout);
    console.log('Python script stderr:', stderr);

    const responseMatch = stdout.match(/AI Response: ([\s\S]+)/);
    const response = responseMatch ? responseMatch[1].trim() : stderr || 'No response generated';
    if (stderr && !responseMatch) console.warn('Falling back to stderr due to parsing failure:', stderr);
    return { response };
  } catch (error: any) {
    console.error('Error executing Python script:', error.message, error.stderr);
    throw new Error(`Python script execution failed: ${error.stderr || error.message}`);
  }
}

// Admin-only PDF upload handler
export const handlePdfUpload: RequestHandler = async (req, res) => {
  try {
    const userId = req.user?.id;
    const file = req.file;

    console.log('Received userId:', userId);
    console.log('Received file:', file);

    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    if (!file) {
      res.status(400).json({ success: false, message: 'PDF file is required.' });
      return;
    }

    const pdfPath = path.resolve(file.path);
    console.log('PDF file path:', pdfPath);

    const inputData: UserInputData = { pdf_path: pdfPath };
    const { response } = await processUserInput(inputData);

    // Clean up uploaded file
    if (fs.existsSync(pdfPath)) fs.unlinkSync(pdfPath);
    console.log('Temporary PDF file cleaned up');

    res.status(201).json({ success: true, message: response });
  } catch (error: any) {
    console.error('Error in handlePdfUpload:', error.message);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ success: false, message: error.message });
  }
};

// User chat message handler
export const handleChatMessage: RequestHandler = async (req, res) => {
  try {
    const userId = req.user?.id;
    const { userMessage, chatId } = req.body;
    const imageFile = req.file;

    console.log('Received userId:', userId);
    console.log('Received userMessage:', userMessage);
    console.log('Received chatId:', chatId);
    console.log('Received image file:', imageFile);

    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    if (!userMessage && !imageFile) {
      res.status(400).json({ success: false, message: 'Text or image input is required.' });
      return;
    }

    let imagePath: string | undefined;
    let imageUrl: string | undefined;

    // Handle image upload to Cloudinary if an image is provided
    if (imageFile) {
      imagePath = path.resolve(imageFile.path);
      const uploadResult = await uploadImage(imagePath);
      imageUrl = uploadResult.secure_url;
      console.log('Image uploaded to Cloudinary:', imageUrl);
    }

    let chatHistory = chatId ? await ChatHistory.findById(chatId) : null;
    if (!chatHistory) {
      console.log('Starting new chat session');
      chatHistory = new ChatHistory({ userId, chat_contents: [] });
    }

    const userMessageId = getNextMessageId(chatHistory.chat_contents);
    console.log('Generated userMessageId:', userMessageId);

    const inputData: UserInputData = { text_input: userMessage, image_path: imagePath };
    const { response: botResponse } = await processUserInput(inputData);
    console.log('Bot response received:', botResponse);

    const botMessageId = userMessageId + 1;

    // Push user message with optional image URL
    if (userMessage || imageUrl) {
      chatHistory.chat_contents.push({
        id: userMessageId,
        sent_by: 'User',
        text_content: userMessage || '',
        timestamp: new Date(),
        image_url: imageUrl,
      });
    }

    chatHistory.chat_contents.push({
      id: botMessageId,
      sent_by: 'Bot',
      text_content: botResponse,
      timestamp: new Date(),
      is_liked: false,
    });

    await chatHistory.save();
    console.log('Chat history saved:', chatHistory._id);

    // Clean up temporary image file after processing
    if (imagePath && fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
      console.log('Temporary image file cleaned up');
    }

    res.status(201).json({ success: true, chatHistory });
  } catch (error: any) {
    console.error('Error in handleChatMessage:', error.message);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ success: false, message: error.message });
  }
};

const getNextMessageId = (chatContents: any[]): number => {
  return chatContents.length > 0 ? chatContents[chatContents.length - 1].id + 1 : 1;
};

// Other controllers remain unchanged
export const getAllChats: RequestHandler = async (req, res) => {
  try {
    const userId = req.user?.id;
    console.log('Fetching all chats for userId:', userId);

    // Fetch chats with chat_name and chat_contents
    const chatHistories = await ChatHistory.find({ userId }).select('chat_name chat_contents');

    // Map the results to include the last message timestamp
    const formattedChatHistories = chatHistories.map(chat => {
      const lastMessage = chat.chat_contents.length > 0
        ? chat.chat_contents.reduce((latest, current) => 
            new Date(latest.timestamp) > new Date(current.timestamp) ? latest : current
          )
        : null;

      return {
        _id: chat._id,
        chat_name: chat.chat_name,
        timestamp: lastMessage ? lastMessage.timestamp : null
      };
    });

    res.status(200).json({ success: true, chatHistories: formattedChatHistories });
  } catch (error: any) {
    console.error('Error in getAllChats:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const getChatHistory: RequestHandler = async (req, res) => {
  try {
    const chatId = req.params.chatId;
    console.log('Fetching chat history for chatId:', chatId);
    const chatHistory = await ChatHistory.findById(chatId).populate('userId', 'name');
    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }
    res.status(200).json({ success: true, chatHistory });
  } catch (error: any) {
    console.error('Error in getChatHistory:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const updateChatName: RequestHandler = async (req, res) => {
  try {
    const { chatId } = req.params;
    const { newChatName } = req.body;
    console.log('Updating chat name for chatId:', chatId, 'to:', newChatName);

    if (!newChatName) {
      console.log('New chat name not provided');
      res.status(400).json({ success: false, message: 'New chat name is required.' });
      return;
    }

    const chatHistory = await ChatHistory.findByIdAndUpdate(
      chatId,
      { chat_name: newChatName },
      { new: true }
    );

    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }

    res.status(200).json({ success: true, chatHistory });
  } catch (error: any) {
    console.error('Error in updateChatName:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const toggleBotResponseLikeStatus: RequestHandler = async (req, res) => {
  try {
    const { chatId, messageId } = req.params;
    console.log('Toggling like status for chatId:', chatId, 'messageId:', messageId);

    const chatHistory = await ChatHistory.findById(chatId);
    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }

    const message = chatHistory.chat_contents.find(
      (content) => content.id === parseInt(messageId) && content.sent_by === 'Bot'
    );

    if (!message) {
      console.log('Bot message not found');
      res.status(404).json({ success: false, message: 'Bot message not found.' });
      return;
    }

    message.is_liked = !message.is_liked;
    await chatHistory.save();
    console.log('Like status updated');
    res.status(200).json({ success: true, message: 'Bot response like status updated.', chatHistory });
  } catch (error: any) {
    console.error('Error in toggleBotResponseLikeStatus:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const deleteChat: RequestHandler = async (req, res) => {
  try {
    const { chatId } = req.params;
    console.log('Deleting chat with chatId:', chatId);

    const chatHistory = await ChatHistory.findByIdAndDelete(chatId);
    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }

    res.status(200).json({ success: true, message: 'Chat deleted successfully.' });
  } catch (error: any) {
    console.error('Error in deleteChat:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};