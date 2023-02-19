uint murmur_hash_11(uint src) {
    const uint M = 0x5bd1e995;
    uint h = 1190494759;
    src *= M;
    src ^= src >> 24;
    src *= M;
    h *= M;
    h ^= src;
    h ^= h >> 13;
    h *= M;
    h ^= h >> 15;

    return h;
}

vec3 murmur_hash_11_color(uint src) {
    const uint hash = murmur_hash_11(src);
    return vec3(float((hash >> 16) & 0xFF), float((hash >> 8) & 0xFF), float(hash & 0xFF)) / 256.0;
}