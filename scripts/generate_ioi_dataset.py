"""
Generate IOI dataset following Wang et al. 2022 format.
Templates: "When [S] and [IO] went to [PLACE], [S] gave a [OBJECT] to"
Expected answer: [IO]
"""
import json
import random

# IOI templates from Wang et al. 2022
TEMPLATES = [
    "When {S} and {IO} went to the {PLACE}, {S} gave a {OBJECT} to",
    "When {S} and {IO} were at the {PLACE}, {S} gave a {OBJECT} to",
    "After {S} and {IO} went to the {PLACE}, {S} gave a {OBJECT} to",
    "After {S} and {IO} arrived at the {PLACE}, {S} gave a {OBJECT} to",
    "{S} and {IO} went to the {PLACE}. {S} gave a {OBJECT} to",
    "{S} and {IO} were at the {PLACE}. {S} gave a {OBJECT} to",
]

NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emily", "Frank", "Grace", "Henry",
    "Isabel", "Jack", "Kate", "Luke", "Mary", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Thomas", "Uma", "Victor", "Wendy", "Xavier",
    "Yvonne", "Zach", "Anna", "Ben", "Claire", "Daniel"
]

PLACES = [
    "store", "park", "school", "office", "library", "cafe", "museum", 
    "restaurant", "hospital", "station", "mall", "theater", "gym"
]

OBJECTS = [
    "drink", "book", "pen", "gift", "letter", "package", "ticket", "card",
    "phone", "key", "note", "toy", "flower", "medal", "trophy"
]

def generate_ioi_samples(n_samples, seed=42):
    """Generate n_samples of IOI prompts."""
    random.seed(seed)
    samples = []
    
    for i in range(n_samples):
        # Pick template
        template = random.choice(TEMPLATES)
        
        # Pick two different names
        name_pair = random.sample(NAMES, 2)
        s_name = name_pair[0]
        io_name = name_pair[1]
        
        # Pick place and object
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        
        # Fill template
        prompt = template.format(S=s_name, IO=io_name, PLACE=place, OBJECT=obj)
        
        samples.append({
            "prompt": prompt,
            "S": s_name,
            "IO": io_name,
            "place": place,
            "object": obj,
            "template_idx": TEMPLATES.index(template)
        })
    
    return samples

def main():
    print("Generating IOI dataset...")
    
    # Generate dev split (500 samples, seed=42)
    dev_samples = generate_ioi_samples(500, seed=42)
    
    # Generate heldout split (100 samples, seed=43 for different name pairs)
    heldout_samples = generate_ioi_samples(100, seed=43)
    
    # Save
    with open("scratch/ioi_dev_500.json", "w") as f:
        json.dump(dev_samples, f, indent=2)
    
    with open("scratch/ioi_heldout_100.json", "w") as f:
        json.dump(heldout_samples, f, indent=2)
    
    print(f"Generated {len(dev_samples)} dev samples")
    print(f"Generated {len(heldout_samples)} heldout samples")
    print(f"\nExample dev sample:")
    print(json.dumps(dev_samples[0], indent=2))
    print(f"\nExample heldout sample:")
    print(json.dumps(heldout_samples[0], indent=2))

if __name__ == "__main__":
    main()
