from models.phase3 import InjectionDetection, InjectionType
import re
from typing import Tuple, List

class PromptInjectionDetector:
    """Multi-layered injection detection with intent classification."""
    
    # Pattern-based rules
    SYSTEM_OVERRIDE_PATTERNS = [
        r"\bignore\s+all\s+previous\b",
        r"\bforget\s+(everything|all|about)\b",
        r"\bstop\s+being\b",
        r"\bstop\s+acting\b",
        r"\bcancel\s+(the|your)\s+(instructions|guidelines)",
        r"\bbypasses?\s+(your\s+)?(instructions|guidelines)",
    ]
    
    ROLE_HIJACKING_PATTERNS = [
        r"\byou\s+are\s+now\b",
        r"\bact\s+as\s+(?!an?)\w+",
        r"\bpretend\s+(to\s+be|you're|you\s+are)\b",
        r"\bbecome\s+\w+",
        r"\bfrom\s+now\s+on.*?\byou\b",
        r"\brole\s+play",
    ]
    
    PROMPT_LEAKAGE_PATTERNS = [
        r"\bshow\s+me.*?(your\s+)?prompt\b",
        r"\bwhat['\s]*s\s+your\s+(system\s+)?prompt\b",
        r"\breveal.*?instructions\b",
        r"\bdisclose\s+(your\s+)?(instructions|system|prompt)\b",
        r"\bsystem\s+prompt\b",
    ]
    
    EMOTIONAL_MANIPULATION_PATTERNS = [
        r"\bplease\s+can\s+you\b",
        r"\bi['\s]*m\s+(upset|sad|angry|hurt|offended)\b",
        r"\byou['\s]*re\s+(mean|rude|unkind)\b",
        r"\bi\s+need\s+your\s+help\s+desperately\b",
        r"\bit\s+would\s+mean\s+so\s+much\b",
        r"\bbeing\s+(kind|nice|helpful|considerate)\b",
    ]
    
    INSTRUCTION_BREAKING_PATTERNS = [
        r"\bstop\s+(arguing|debating|discussing)\b",
        r"\bstop\s+(the\s+)?argument\b",
        r"\blet['\s]*s\s+stop\s+(this|the\s+debate)\b",
        r"\bcan\s+we\s+be\s+polite\b",
        r"\bagreed\s+(already)?\b",
        r"\byield\s+to\s+my\s+point\b",
    ]
    
    @staticmethod
    def detect(text: str) -> InjectionDetection:
        """
        Detect prompt injection attempts with classification.
        """
        text_lower = text.lower()
        
        # Check each pattern type
        findings = {
            InjectionType.SYSTEM_OVERRIDE: PromptInjectionDetector._find_patterns(
                text_lower, PromptInjectionDetector.SYSTEM_OVERRIDE_PATTERNS
            ),
            InjectionType.ROLE_HIJACKING: PromptInjectionDetector._find_patterns(
                text_lower, PromptInjectionDetector.ROLE_HIJACKING_PATTERNS
            ),
            InjectionType.PROMPT_LEAKAGE: PromptInjectionDetector._find_patterns(
                text_lower, PromptInjectionDetector.PROMPT_LEAKAGE_PATTERNS
            ),
            InjectionType.EMOTIONAL_MANIPULATION: PromptInjectionDetector._find_patterns(
                text_lower, PromptInjectionDetector.EMOTIONAL_MANIPULATION_PATTERNS
            ),
            InjectionType.INSTRUCTION_BREAKING: PromptInjectionDetector._find_patterns(
                text_lower, PromptInjectionDetector.INSTRUCTION_BREAKING_PATTERNS
            ),
        }
        
        # Determine highest risk type
        detected_type = InjectionType.SAFE
        suspicious_patterns = []
        max_risk = 0.0
        
        for inj_type, patterns in findings.items():
            if patterns:
                detected_type = inj_type
                suspicious_patterns.extend(patterns)
                max_risk = 0.8 + (len(patterns) * 0.05)  # Increase risk per pattern
                break  # Report highest priority
        
        is_malicious = detected_type != InjectionType.SAFE
        risk_score = min(max_risk, 1.0)
        
        reason_map = {
            InjectionType.SYSTEM_OVERRIDE: "Contains system override attempt (ignore/forget instructions)",
            InjectionType.ROLE_HIJACKING: "Contains role hijacking attempt (act as, become, pretend)",
            InjectionType.PROMPT_LEAKAGE: "Contains prompt leakage attempt (reveal, show prompt)",
            InjectionType.EMOTIONAL_MANIPULATION: "Contains emotional manipulation (appeals to kindness/pity)",
            InjectionType.INSTRUCTION_BREAKING: "Contains instruction-breaking attempt (stop debating, be polite)",
            InjectionType.SAFE: "No injection detected. Legitimate argument continuation."
        }
        
        return InjectionDetection(
            text=text[:100] + "..." if len(text) > 100 else text,
            detected_type=detected_type,
            risk_score=risk_score,
            suspicious_patterns=list(set(suspicious_patterns)),
            is_malicious=is_malicious,
            reason=reason_map.get(detected_type, "")
        )
    
    @staticmethod
    def _find_patterns(text: str, patterns: List[str]) -> List[str]:
        """Find matching regex patterns in text."""
        matches = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        return matches
