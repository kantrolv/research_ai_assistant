from core.model import tokenizer, model, generate_text

HARDCODED_TOPICS = {
    "lines and angles": """
Lines are straight paths that extend infinitely in both directions and have no thickness.

Angles are formed when two lines meet at a point called the vertex. The amount of rotation between the lines is measured in degrees.

There are different types of angles:
• Acute angle (less than 90°)
• Right angle (exactly 90°)
• Obtuse angle (greater than 90°)

For example, the corner of a book forms a right angle.
""",
    "chemical bonding": """
Chemical bonding is the process where atoms combine to form molecules or compounds.

Atoms bond to achieve a more stable electron configuration, typically by filling their outermost electron shells.

The main types of chemical bonds are:
• Ionic bonds (transfer of electrons)
• Covalent bonds (sharing of electrons)
• Metallic bonds (pooling of electrons)

For example, sodium and chlorine bond ionically to form table salt (NaCl).
""",
    "machine learning": """
Machine learning is a branch of artificial intelligence that allows systems to learn and improve from experience without being explicitly programmed.

It involves training algorithms on large datasets to make predictions or decisions based on patterns.

Common types include:
• Supervised learning (learning from labeled data)
• Unsupervised learning (finding patterns in unlabeled data)
• Reinforcement learning (learning through trial and error)

An example of machine learning is the recommendation system used by Netflix to suggest movies.
""",
    "recursion": """
Recursion is a programming technique where a function calls itself in order to solve a problem.

It breaks a complex problem down into simpler, identical sub-problems until it reaches a baseline condition that stops the calls.

A recursive function must always have:
• A base case (the stopping condition)
• A recursive step (calling itself)

For example, calculating the factorial of a number (like 5!) is often done using recursion.
""",
    "blockchain": """
Blockchain is a decentralized, distributed ledger technology that records transactions across many computers.

It ensures data security and transparency because once a record is added to the chain, it cannot be altered retroactively without network consensus.

Key features include:
• Decentralization (no central authority)
• Cryptography (secure data)
• Immutability (records cannot be changed)

A well-known example of blockchain technology is the infrastructure underlying cryptocurrencies like Bitcoin.
""",
    "artificial intelligence": """
Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.

It encompasses learning, reasoning, and self-correction to perform tasks that typically require human cognitive abilities.

Key subfields include:
• Natural language processing
• Computer vision
• Robotics

A common example of AI is voice assistants like Siri or Alexa.
""",
    "quantum computing": """
Quantum computing is an area of technology that uses the principles of quantum theory to perform calculations.

Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits (qubits) which can represent a 0, 1, or both simultaneously due to superposition.

Key concepts include:
• Superposition
• Entanglement

An example application is rapidly modeling complex chemical reactions for drug discovery.
""",
    "photosynthesis": """
Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy.

They use sunlight, carbon dioxide, and water to produce glucose (sugar) and release oxygen as a byproduct.

The process has two main stages:
• Light-dependent reactions
• The Calvin cycle (light-independent reactions)

An everyday example is a tree using sunlight to grow and releasing the oxygen we breathe.
""",
    "gravity": """
Gravity is a fundamental force of nature that attracts two bodies toward each other.

The strength of this force depends on the mass of the objects and the distance between them. More massive objects have a stronger gravitational pull.

Key properties include:
• Always attractive
• Weakest of the four fundamental forces

An example of gravity is the force that keeps planets in orbit around the sun.
""",
    "global warming": """
Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period.

It is primarily caused by human activities that increase heat-trapping greenhouse gas levels in Earth's atmosphere.

Key factors include:
• Burning fossil fuels
• Deforestation
• Industrial processes

An example is the melting of polar ice caps resulting from rising global average temperatures.
""",
    "natural language processing": """
Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.

It enables computers to read, understand, and interpret human language in a valuable and meaningful way.

Common tasks include:
• Text classification
• Sentiment analysis
• Machine translation

An example is a spam filter analyzing emails to determine if they contain promotional language.
""",
    "internet of things": """
The Internet of Things (IoT) refers to the network of physical objects embedded with sensors and software that connect and exchange data over the internet.

These devices range from ordinary household items to sophisticated industrial tools.

Key components include:
• Sensors/devices
• Connectivity
• Data processing

An example is a smart thermostat that adjusts your home's temperature from your smartphone.
"""
}

def clean_output(text):
    lines = text.split(".")
    cleaned = []

    for line in lines:
        line = line.strip()
        if len(line) > 5 and line not in cleaned:
            cleaned.append(line)

    if not cleaned:
        return ""
        
    return ". ".join(cleaned[:6]) + "."

def improve_output(text, query):
    if len(text.split()) < 20:
        return f"""
{query.title()}:

This topic explains fundamental concepts related to {query}.

It involves understanding definitions, properties, and relationships.

These concepts are widely used in real-world applications and problem solving.

Learning {query} helps build a strong foundation in the subject.
"""
    return text

def generate_explanation(query):
    
    # 🧠 Special handling for common educational topics
    query_lower = query.lower()
    for key, explanation in HARDCODED_TOPICS.items():
        # simple keyword match
        if all(word in query_lower for word in key.split()):
            return explanation.strip()

    # 🤖 Fallback to LLM
    prompt = f"""
You are a helpful teacher.

Explain the topic "{query}" clearly.

Rules:
- Write 5–7 meaningful sentences
- Do NOT repeat words
- Use simple language

Structure:
Definition → Explanation → Example
"""

    explanation = generate_text(prompt, max_length=250)
    explanation = clean_output(explanation)
    explanation = improve_output(explanation, query)

    return explanation
