import mongoose, { Schema, Document } from 'mongoose';

export interface IUser extends Document {
  email: string;
  password: string;
  name: string;
  role: string;
  profileImage?: string;
  language?: string;
  fcmToken?: string; // Add FCM token field
  birthday?: Date; // Add birthday field for Task 2
}

const UserSchema: Schema = new Schema(
  {
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    name: { type: String, required: true },
    role: { type: String, enum: ['admin', 'user'], default: 'user' },
    profileImage: { type: String },
    language: { type: String, default: null, nullable: true },
    fcmToken: { type: String, default: null }, // Store FCM token
    birthday: { type: Date, default: null }, // Store birthday
  },
  { timestamps: true }
);

export const User = mongoose.model<IUser>('User', UserSchema);