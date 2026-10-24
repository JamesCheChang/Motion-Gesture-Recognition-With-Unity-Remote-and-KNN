# reconstruct_hex.py

def reconstruct_hex_from_fragments(fragments):
    """
    Concatenates bit strings from fragments dictionary and converts to Hex.
    """
    # 1. Concatenate in order m1 to m8
    # Note: m8 is special (reversed)
    
    full_binary = (
        fragments['m1_f'] +
        fragments['m2_f'] +
        fragments['m3_f'] +
        fragments['m4_f'] +
        fragments['m5_f'] +
        fragments['m6_f'] +
        fragments['m7_f'] +
        fragments['m8_f'][::-1] # Reverse m8
    )
    
    # 2. Convert Binary String to Hex
    # int(str, 2) converts binary string to integer
    # hex() converts integer to hex string (e.g., '0x1a...')
    # [2:] removes the '0x' prefix
    hex_val = hex(int(full_binary, 2))[2:].upper()
    
    return hex_val

if __name__ == "__main__":
    # Define Inputs
    fragments = {
        'm1_f': '1010101001010101',
        'm2_f': '0000000000000000',
        'm3_f': '1110110110110111',
        'm4_f': '0001001001001000',
        'm5_f': '0001001001001111',
        'm6_f': '1110110110110000',
        'm7_f': '0000111100001010',
        'm8_f': '1111000011110101' 
    }
    
    result = reconstruct_hex_from_fragments(fragments)
    print(f"The reconstructed hex value is: {result}")
