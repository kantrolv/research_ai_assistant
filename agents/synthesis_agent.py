def generate_explanation(query):
    q = query.lower()

    if "chemical bonding" in q:
        return """
Chemical bonding is the force that holds atoms together to form molecules and compounds.

Atoms bond because they seek stable electron configurations in their outer shells.

There are three main types of chemical bonds:

• Ionic bonds – electrons transfer between atoms.
• Covalent bonds – atoms share electrons.
• Metallic bonds – electrons move freely among metal atoms.

Chemical bonding determines the structure, stability, and properties of substances.
"""

    return f"This section explains the concept of {query}."
