import { describe, it, expect, vi, beforeEach } from 'vitest';
import { transcribeAudio, summarizeTranscript, identifySpeakers, exportTxt, exportPdf } from '../api';

// Mock the global fetch function
global.fetch = vi.fn();

describe('API functions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('transcribeAudio', () => {
    it('returns JSON data on success', async () => {
      /* Proves that a successful fetch call returns the parsed JSON response. */
      const mockResponse = { transcript: 'test text', language: 'en', duration: 10, segments: [] };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const file = new File([''], 'test.wav', { type: 'audio/wav' });
      const result = await transcribeAudio(file, 'groq', 'fake-key');

      expect(fetch).toHaveBeenCalledTimes(1);
      const urlCall = fetch.mock.calls[0][0];
      expect(urlCall).toContain('/api/transcribe');
      expect(result).toEqual(mockResponse);
    });

    it('throws an error on failure', async () => {
      /* Proves that an unsuccessful fetch call throws an Error with the details. */
      fetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'API Error' }),
      });

      const file = new File([''], 'test.wav', { type: 'audio/wav' });
      await expect(transcribeAudio(file, 'groq', 'fake-key')).rejects.toThrow('API Error');
    });
  });

  describe('summarizeTranscript', () => {
    it('returns JSON data on success', async () => {
      /* Proves that a successful fetch call returns the parsed JSON summary data. */
      const mockResponse = { summary: 'summary text', action_items: ['item 1'] };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await summarizeTranscript('some transcript', 'Brief', 'model', 'fake-key');

      expect(fetch).toHaveBeenCalledTimes(1);
      const urlCall = fetch.mock.calls[0][0];
      expect(urlCall).toContain('/api/summarize');
      expect(result).toEqual(mockResponse);
    });

    it('throws an error on failure', async () => {
      /* Proves that an unsuccessful fetch call throws an Error. */
      fetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'Summarize Error' }),
      });

      await expect(summarizeTranscript('some transcript', 'Brief', 'model', 'fake-key')).rejects.toThrow('Summarize Error');
    });
  });
});
