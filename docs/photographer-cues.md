# Photographer Cue Design

## Purpose

The cue layer translates measured analysis into short spoken instructions. It is not a replacement for technical analysis and it must not become a generic "best pose" generator.

## Output layers

```text
Observed measurements
    ↓
Professional analysis
    ↓
Goal-oriented pose guidance
    ↓
Speakable cue
```

The same result should remain inspectable at every layer.

## Cue rules

1. Prefer one physical change at a time.
2. Describe the body part and direction when possible.
3. Explain why in the technical panel, but keep spoken cues short.
4. Do not force the current action category to remain unchanged or to become a specific next action.
5. Respect composition: actions can open toward negative space or away from the crowded edge.
6. Respect framing: tight frames require smaller movement amplitudes.
7. Preserve intentional asymmetry instead of automatically correcting it.
8. Use conservative language when landmark confidence is low.

## Examples

### Stiff symmetric standing

Technical reason: low knee-angle difference, narrow stance, small ankle-height difference.

Cue: “重心放到一条腿上，另一条腿放松一点。”

### Arms touching the torso

Technical reason: both elbows near extension while hands are not raised; silhouette separation is weak.

Cue: “一只手离开身体一点，肘部别夹死。”

### Subject placed near frame edge

Technical reason: body occupies a lateral edge region and useful negative space lies toward the opposite side.

Cue: “动作往你的另一边打开一点。”

## Future cue modes

### Concise
One sentence, optimized for speaking while shooting.

### Normal
One primary command plus one follow-up and a short reason.

### Technical
Command plus measured basis, useful for photographers learning from the system.

The modes should consume the same structured cue data rather than maintaining separate rule engines.
