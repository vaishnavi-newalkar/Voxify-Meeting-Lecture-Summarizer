// import React from 'react';fireEvent, waitFor
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';
// import * as api from '../api';
import { SAMPLE_TRANSCRIPT } from '../constants';

// Mock the API calls
vi.mock('../api', () => ({
  transcribeAudio: vi.fn(),
  summarizeTranscript: vi.fn(),
  identifySpeakers: vi.fn(),
}));

describe('App Component', () => {
  let alertMock;

  beforeEach(() => {
    vi.clearAllMocks();
    alertMock = vi.spyOn(window, 'alert').mockImplementation(() => {});
    
    // Mock scrollIntoView for ResultsPanel
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it('renders without crashing and displays the hero section', () => {
    /* Proves that the main App component mounts successfully. */
    render(<App />);
    expect(screen.getByText(/Meeting notes that feel/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Paste your meeting transcript here/i)).toBeInTheDocument();
  });

  it('populates the transcript text area when "Try a sample" is clicked', async () => {
    /* Proves that clicking the sample button updates the textarea with sample text. */
    render(<App />);
    const sampleBtn = screen.getByRole('button', { name: /Try a sample/i });
    
    await userEvent.click(sampleBtn);
    
    const textarea = screen.getByPlaceholderText(/Paste your meeting transcript here/i);
    expect(textarea.value).toBe(SAMPLE_TRANSCRIPT);
  });

  it('shows an error if the uploaded file is too large (>25MB)', async () => {
    /* Proves that file size validation works and blocks files over 25MB. */
    render(<App />);
    
    // Find the hidden file input
    const fileInput = document.querySelector('input[type="file"]');
    
    // Create a mock file larger than 25MB (25 * 1024 * 1024 bytes)
    const largeBlob = new Blob(['a'.repeat(26 * 1024 * 1024)], { type: 'audio/wav' });
    const largeFile = new File([largeBlob], 'large.wav', { type: 'audio/wav' });
    
    // We have to mock size property because JSDOM might not calculate Blob size correctly for huge strings if not handled well, but it should work.
    Object.defineProperty(largeFile, 'size', { value: 26 * 1024 * 1024 });

    await userEvent.upload(fileInput, largeFile);
    
    // Check if the warning is displayed
    expect(screen.getByText(/File exceeds.*limit/i)).toBeInTheDocument();
    
    // Summarize button should be disabled
    const summarizeBtn = screen.getByRole('button', { name: /File too large/i });
    expect(summarizeBtn).toBeDisabled();
  });

  it('alerts the user if summarize is clicked without providing an API key', async () => {
    /* Proves that the app prevents processing if the API key is missing. */
    render(<App />);
    
    // Enter some transcript so it doesn't fail the "empty text" check
    const textarea = screen.getByPlaceholderText(/Paste your meeting transcript here/i);
    await userEvent.type(textarea, 'Some fake transcript text.');
    
    const summarizeBtn = screen.getByRole('button', { name: /Summarize/i });
    await userEvent.click(summarizeBtn);
    
    expect(alertMock).toHaveBeenCalledWith('Please enter your API key in Settings first.');
  });
});
