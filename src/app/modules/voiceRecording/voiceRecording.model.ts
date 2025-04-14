import mongoose, { Schema, Document } from 'mongoose';

export interface IVoiceRecording extends Document {
  user: mongoose.Types.ObjectId;
  title: string;
  relation: string;
  imageUrl?: string;
  audioUrl: string;
  clonedVoiceId?: string;
  personalizationData: {
    lovedOneName: string;
    lovedOneBirthday: string;
    userBirthday: string;
    distinctGreeting: string;
    distinctGoodbye: string;
    signaturePhrase: string;
    favoriteSong: string;
    favoriteTopic: string;
    nicknameForUser: string;
    nicknameForLovedOne: string;
    customQuestions?: { question: string; answer: string }[];
  };
}

const VoiceRecordingSchema: Schema = new Schema(
  {
    user: { type: mongoose.Types.ObjectId, ref: 'User', required: true },
    title: { type: String, required: true },
    relation: { type: String, required: true },
    imageUrl: { type: String },
    audioUrl: { type: String, required: true },
    clonedVoiceId: { type: String },
    personalizationData: {
      lovedOneName: { type: String},
      lovedOneBirthday: { type: String },
      userBirthday: { type: String },
      distinctGreeting: { type: String},
      distinctGoodbye: { type: String},
      signaturePhrase: { type: String},
      favoriteSong: { type: String },
      favoriteTopic: { type: String },
      nicknameForUser: { type: String},
      nicknameForLovedOne: { type: String },
      customQuestions: [
        {
          question: { type: String },
          answer: { type: String },
        },
      ],
    },
  },
  { timestamps: true }
);

export const VoiceRecording = mongoose.model<IVoiceRecording>('VoiceRecording', VoiceRecordingSchema);