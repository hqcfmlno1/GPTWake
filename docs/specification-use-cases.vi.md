# GPTWake — Usage and Anchor Management Specification

## 1. Purpose and Practical Value

GPTWake lets users say “Hey GPT” to start a ChatGPT Voice conversation, reducing the need to find tabs, switch windows, and use a mouse or keyboard. The system manages an anchor that determines where subsequent Voice calls start.

- **Studying on a laptop with a small screen:** students reading PDFs, watching lectures, or writing code can ask questions by voice without manually finding the ChatGPT tab or splitting the screen. A fixed URL directs subsequent questions to the same chat for a course, assignment, or project.
- **Supporting users who have difficulty operating input devices:** wake word activation reduces reliance on manual input when starting a conversation and may assist users with motor impairments. The MVP does not promise completely hands-free operation; signing in, granting permissions, and resolving some errors may still require the user or an assistant to take action.
- **Asking questions while working:** users can call the assistant while working in another application, reducing the steps required to start and continue a conversation.

The benefits to evaluate are fewer manual actions, less time to start Voice, and fewer workflow interruptions. Continuing to view study materials while talking depends on background Voice behavior and the activation process in supported browsers; the system must not assume that the screen will always remain unchanged.

## 2. MVP Scope and Limitations

GPTWake captures audio and detects the wake word locally. After a valid detection, it pauses the detector, resolves the anchor, opens or brings the appropriate tab to the foreground when necessary, and activates Voice through UI controls. Once the call has actually ended, the system resets the detector and resumes listening.

- The primary workflow uses ChatGPT Web and does not require ChatGPT Desktop, an API key, or the OpenAI Realtime API. Free accounts are intended users; Voice availability and usage limits depend on the service, account, and browser at runtime.
- The system provides two anchor configurations: **default** and **a specific chat URL**.
- The MVP has no separate model for detecting a cancellation command. Users may ask ChatGPT to end the call or use its existing end-call control; GPTWake must verify the actual call state.
- A fixed URL preserves the conversation destination, but does not automatically guarantee that responses remain on topic. Users need to prepare the chat with appropriate instructions or materials.
- GPTWake does not automatically read the PDF, screen, or application the user is viewing. Switching chats does not automatically transfer history from the previous chat to the new one.

## 3. Definitions and Prerequisites

- **Selected tab:** the tab that will be displayed when its browser window is brought into view. The window may be visible, minimized, or behind another application. The window does not need to have operating system focus.
- **Open but unselected ChatGPT tab:** a tab that still exists in a window while another tab is selected. This tab is not a candidate when choosing an anchor in default mode.
- **Anchor:** the destination used to start Voice. In default mode, the anchor is associated with a tab; in specific URL mode, it is associated with a fixed chat.
- **Chat:** a ChatGPT conversation with its own identifier or URL, potentially containing multiple Voice calls.
- **Voice call:** one entry into Voice mode until that mode has actually ended.
- **Boot session:** one operating system boot session. Shutting down and starting the computer again, or restarting it, begins a new session; resuming the same operating system session from sleep or hibernation does not.
- **Control scope:** the configured, supported browsers and profiles that GPTWake is authorized to control. Support for every browser on the computer must not be assumed.

Users must grant microphone access to GPTWake and ChatGPT Web, sign in to the correct account in the selected profile, have a network connection, and be eligible to use Voice. Authentication or permissions may need to be renewed. Users may configure GPTWake to start automatically with the operating system.

The configuration must specify the default browser and profile for creating new tabs. Specific URL mode must identify a profile that can access the chat. Identifying multiple windows and their interaction order must rely on information actually available through the browser integration.

## 4. Default Mode — Anchor Selection

**Principle:** select an anchor only when no valid anchor exists. Do not select it again based on the window the user is viewing at every wake event.

When a wake event requires anchor selection, the system captures the browser state before bringing any window or tab to the foreground, then applies these rules:

1. Find windows within the control scope whose selected tab is ChatGPT, including windows that are minimized or behind another application.
2. If there is exactly one candidate, use that tab as the anchor.
3. If there are multiple candidates, use the tab belonging to the window the user interacted with most recently. GPTWake's own window-switching actions must not update this user interaction priority.
4. If there are no candidates, create a new ChatGPT tab with a new chat in the default browser and profile, and use the new tab as the anchor. This also applies when open ChatGPT tabs exist but none is selected.
5. Bring the anchor tab to the foreground when needed and start Voice in the chat currently displayed in that tab.

If the interaction order between multiple candidates cannot be determined, create a new tab in the default browser rather than choosing a chat arbitrarily.

For example, if Edge has a ChatGPT tab selected while the user is working in VS Code, that ChatGPT tab remains a candidate. If Edge has a PDF tab selected and the ChatGPT tab is unselected, the ChatGPT tab is not a candidate; if there are no other candidates, GPTWake creates a new tab.

## 5. Default Mode — Maintaining and Replacing the Anchor

- Ending Voice does not clear the anchor. The next wake event resumes Voice in the anchor tab.
- Switching to another tab, window, or application does not change the anchor. Minimizing the window does not change it either.
- If the user navigates from chat A to chat B within the anchor tab, the next wake event uses chat B. Default mode preserves the tab, rather than permanently locking onto chat A's identifier.
- If the anchor tab is closed or navigated away from ChatGPT, the anchor becomes invalid. At the next wake event, the system selects an anchor again using Section 4; it must not immediately open a tab or start Voice merely because the anchor became invalid.
- Closing the browser containing the anchor is treated as closing the anchor tab. Default mode is not required to reopen the previous anchor's URL.
- Restarting GPTWake alone within the same boot session requires checking the saved tab: reuse it if it still exists and is valid; otherwise clear the anchor and select another at the next wake event. Voice state must be checked before enabling the detector.
- After a shutdown and startup or a computer restart, discard the previous session's anchor and select again at the first wake event. The browser may restore an old chat, which may be selected again if it satisfies the rules; a new chat is not mandatory after every computer startup.
- Resuming the same session from sleep or hibernation preserves the anchor if its tab remains valid; tab and Voice states must be checked again.

## 6. Specific Chat URL Mode — Fixed Destination

**Principle:** the user supplies the URL of a chat that supports continued conversation. The system always returns to that chat and must not automatically substitute another one.

When a wake word is detected:

1. Resolve the target chat from the configured URL and use the specified browser and profile.
2. If a tab in that profile already displays the target chat, use it, even if it is unselected or its window is minimized.
3. If multiple tabs display the target chat, prefer the tab the user used most recently. If their order cannot be determined, select one using a stable ordering without creating duplicate calls.
4. If no tab displays the target chat, open the configured URL in a new tab.
5. Confirm that the correct chat is open, bring the tab to the foreground when needed, and activate Voice.

Other ChatGPT tabs must not be used as substitutes. If the user closes the tab, the next wake event reopens the same URL. If the user navigates to another chat within the same tab, the next wake event still returns to the configured chat.

The URL and profile configuration persist across shutdowns and computer restarts. The system starts Voice only after a wake word; application startup alone must not start a call.

If the URL is invalid, the chat has been deleted, the account lacks access, or the page redirects to a different chat, report an error. Do not automatically create a replacement chat or switch to default mode. A homepage URL or a shared viewing link must not be treated as a fixed chat URL that supports continued Voice unless that capability has been verified.

## 7. Voice Call Workflow

**Preconditions:** GPTWake is listening, the configuration is valid, and there is no active call that the system has not accounted for.

1. The user says “Hey GPT.” The system accepts exactly one wake event and immediately stops wake word detection.
2. Resolve the destination according to default or specific URL mode; check the tab and chat again before interacting with them.
3. Open the page or bring the appropriate tab to the foreground if needed; verify authentication, microphone permissions, and Voice availability.
4. Activate Voice. Mark the system as conversing only after confirming that the UI has entered a call.
5. Throughout the call, do not process wake words or accumulate queued audio for detection after the call.
6. The user converses and ends the call through ChatGPT's existing capabilities or UI controls.
7. Confirm that the call has actually ended; clear audio buffers, smoothing state, and stale inference results; resume listening.

If Voice is already active at the destination, do not click to start another call. Synchronize with the existing call and keep the detector paused until its end is confirmed.

## 8. State Management and Configuration Changes

Main states:

```text
LISTENING -> RESOLVING_ANCHOR -> OPENING_WEB -> STARTING_VOICE
STARTING_VOICE -> CONVERSING -> REARMING -> LISTENING
ERROR_OR_UNKNOWN -> RECOVERING -> REARMING -> LISTENING
```

- Only LISTENING accepts wake words. Overlapping audio windows or delayed inference results must not trigger additional activation workflows.
- Confirm the end of a call only after the call was known to be active and the UI has remained in a non-call state consistently. Handle the end event only once per call.
- Silence, ChatGPT finishing a response, window minimization, or a temporarily missing control is insufficient evidence that the call has ended.
- Losing communication with the controller does not mean Voice has stopped. When the state is unknown, remain in recovery and do not re-enable the detector.
- Changing the mode or URL during a call does not redirect the current call; the new configuration takes effect at the next wake event. Switching to default mode clears the previous default anchor and selects again using the state at that wake event.
- The controller must verify UI recognition, tab state, and waiting periods in supported environments. Selector and timeout details belong to the implementation design.

## 9. Exceptions and Recovery

- **False detection:** follow the normal Voice workflow; the user ends the call, and the system confirms the end before listening again. Do not add a separate cancellation-command detection model to the MVP.
- **Not signed in or missing microphone permission:** explain the required action. Do not treat Voice as active; if the user cancels or the timeout expires, end the attempt in a controlled manner.
- **Network failure, exhausted Voice allowance, page loading failure, or missing controls:** use a finite timeout, report the cause, and avoid indefinitely opening tabs or repeatedly clicking Voice.
- **Tab closed during a call:** confirm that the call context has closed before resuming listening. Default mode clears the anchor; fixed URL mode preserves its configuration for reopening at the next wake event.
- **Inaccessible chat:** fixed URL mode reports an error and preserves its destination. Default mode must not treat loading failures or lost connectivity as reasons to create chats repeatedly; the user may close the failed tab so the next wake event selects again.
- **Configured browser or profile unavailable:** report that action is required; do not silently use another profile.
- **Window minimization:** do not automatically end Voice. Background operation must be verified. If activation requires bringing ChatGPT to the foreground, the interface must communicate that behavior; uninterrupted screen use must not be promised without verification.

## 10. Conversation History and Context

Default mode prioritizes flexibility around an anchored tab; fixed URL mode prioritizes continuing the same chat for a topic. Closing a tab does not automatically delete ChatGPT history, but default mode may select a different chat at the next wake event.

Chat A and chat B have separate histories. GPTWake does not automatically copy or merge content between chats. Continuing the same chat preserves its history, but how much previous content ChatGPT uses still depends on the service's context mechanism. A fixed URL must not be described as guaranteeing unlimited memory.

## 11. Functional Acceptance Criteria

1. A single “Hey GPT” utterance spanning multiple audio windows produces only one Voice activation workflow.
2. In default mode without an anchor, if one window has ChatGPT selected but is minimized or behind VS Code, use that ChatGPT tab.
3. In default mode without an anchor, if multiple windows have ChatGPT selected, choose the window the user interacted with most recently before GPTWake's own actions.
4. In default mode without an anchor, if only unselected ChatGPT tabs exist, create a new tab instead of reusing them.
5. In default mode without an anchor or any ChatGPT tabs, create a new tab in the default browser and profile.
6. In default mode with anchor tab A, switching to tab B or another application before a wake event still results in using tab A.
7. In default mode, closing the anchor tab and then waking causes selection to run again; reopening the old chat is not mandatory.
8. In default mode, navigating from chat A to B within the anchor tab causes the next wake event to use B.
9. After a computer restart, default mode selects again based on browser state; creating a new chat is not mandatory.
10. Restarting GPTWake alone or resuming the same session from sleep preserves a valid default anchor and checks Voice before listening.
11. In fixed URL mode, an existing tab displaying the target chat is used even if it is unselected.
12. In fixed URL mode, if no matching tab exists, the tab was closed, or the computer has just restarted, open the configured URL.
13. In fixed URL mode, a deleted chat, insufficient access, or an invalid URL produces an error without switching chats or automatically creating a replacement.
14. If Voice is already active at the destination, do not create another call or enable the detector.
15. While opening a page or conversing, audio containing “Hey GPT” does not create another call; duplicate end events do not trigger reactivation.
16. Silence, ChatGPT finishing a response, or window minimization does not resume listening while the call remains open.
17. Authentication, permission, network, and usage-limit errors are reported, use finite timeouts, and do not cause repeated tab creation.
18. Changing configuration during a call does not interrupt it; the next wake event uses the new configuration.

## 12. Evaluating Practical Value

Test document reading, programming, and study-related question answering. Compare manual operation with GPTWake using the same environment, account, and tasks.

- Record the number of mouse and keyboard actions and the time from wanting to ask a question until Voice is ready.
- Record how often users must find tabs or switch windows, and the interruptions experienced on a small laptop screen.
- Check whether follow-up questions return to the configured chat, particularly with a URL dedicated to a course or project.
- Evaluate ease of use, false activations, and steps that still require manual assistance. Do not generalize the results to support for every type of disability without appropriate testing.
- Verify background Voice operation and the ability to continue viewing materials after activation on each supported browser and operating system. Report experimental results rather than treating these capabilities as already proven.
