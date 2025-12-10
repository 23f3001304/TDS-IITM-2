"""
Agent Package - Quiz Solving Agent with Tools
"""
# Import models first
from app.agent.models import QuizDependencies, QuizAnswer, QuizContext

# Import prompts and agents
from app.agent.prompts import quiz_agent, guidance_agent, SYSTEM_PROMPT, GUIDANCE_PROMPT

# Import tools to register them with quiz_agent
from app.agent import tools

# Import solver
from app.agent.solver import QuizSolver

# Exports
agent = QuizSolver()
QuizAgent = QuizSolver  # Alias for backwards compatibility

__all__ = [
    'QuizDependencies',
    'QuizAnswer', 
    'QuizContext',
    'quiz_agent',
    'guidance_agent',
    'QuizSolver',
    'QuizAgent',
    'agent',
]
