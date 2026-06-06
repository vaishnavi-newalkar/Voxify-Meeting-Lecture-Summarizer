export const SAMPLE_TRANSCRIPT = `Good morning everyone. Let's get started with our weekly product sync. I want to cover three main topics today: the Q3 roadmap update, the customer feedback from the beta launch, and our hiring plan for the engineering team.

Starting with the Q3 roadmap — we're on track for the mobile app release in August. The design team has finalized the new onboarding flow, and engineering is about two weeks into the sprint. Sarah, can you give us an update on the backend API migration?

Sure. The API migration is about 70% complete. We've moved the authentication and user management endpoints to the new architecture. The remaining work is the analytics pipeline and the notification service. I estimate we'll be done by mid-July, which gives us enough buffer before the August launch.

That sounds great. Now regarding the beta feedback — we've received over 200 responses from our early access users. The NPS score is 72, which is significantly higher than our target of 60. The most requested feature is real-time collaboration, followed by better export options and custom templates.

I think we should prioritize real-time collaboration for v2. It aligns with our enterprise push and several potential customers have listed it as a must-have requirement.

Agreed. Let's add that to the Q4 roadmap. Mike, can you draft a technical proposal for the collaboration feature by next Friday?

Absolutely. I'll also loop in the security team since real-time features will need WebSocket connections and we need to make sure our infrastructure can handle the increased load.

Perfect. Last topic — hiring. We have budget approval for three senior engineers and one product designer. I'd like to have offers out by end of July. HR has already started sourcing candidates. Let's make sure we're all available for the interview panels.

One more thing — the board presentation is next Thursday. I'll need the updated metrics dashboard by Wednesday. Can the analytics team prioritize that?

Yes, we'll have it ready by Tuesday evening so there's time for review.

Great. Let's wrap up. Action items: Sarah continues the API migration, Mike drafts the collaboration proposal, analytics team delivers the dashboard by Tuesday, and everyone blocks time for interviews. Same time next week. Thanks everyone.`;

export const LLM_OPTIONS = [
  { label: "Groq — llama-3.3-70b (Recommended)", value: "llama-3.3-70b-versatile" },
  { label: "Groq — llama-3.1-8b-instant (Fast)", value: "llama-3.1-8b-instant" },
  { label: "Groq — llama3-70b-8192 (Balanced)", value: "llama3-70b-8192" },
];

export const LENGTH_OPTIONS = [
  "Brief (3–5 points)",
  "Standard (5–8 points)",
  "Detailed (full breakdown)",
];
