// Inflected forms of lexicon words (2026-11-16). FluidAudio 0.18.2's Swift port of Misaki's English frontend handles
// the possessive "'s" but not Misaki's other inflection rules, so a word like "Ravens" (only "raven" is in the lexicon)
// went to the BART G2P model, which guesses from spelling and often mispronounces. This ports Misaki's rules
// (misaki/en.py 0.7.4: Lexicon.stem_s, stem_ed, stem_ing with _s, _ed, _ing, and apply_stress for the 0.5 stress they
// pass): a word missing from the lexicon whose stem is in it gets the stem's phonemes plus the suffix. The results go
// to FluidAudio through its own hook (KokoroAneManager.setEnglishCustomLexicon), so nothing in FluidAudio changes.

import Foundation

struct Inflections {
    let lower: [String: String]
    let caseSensitive: [String: String]

    static let usTaus: Set<Character> = Set("AIOWYiuæɑəɛɪɹʊʌ")       // misaki US_TAUS
    static let vowels: Set<Character> = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")   // misaki VOWELS
    static let stresses: Set<Character> = ["ˈ", "ˌ"]

    /// The lexicon FluidAudio loads (`<models>/kokoro/us_lexicon_cache.json`), with each entry's phonemes filtered to
    /// the chain's vocab as FluidAudio does (LexiconAssetCache.ensureLoaded).
    static func load(models: URL) throws -> Inflections {
        struct Payload: Decodable {
            let lower: [String: [String]]
            let caseSensitive: [String: [String]]
        }
        let lexicon = models.appendingPathComponent("kokoro/us_lexicon_cache.json")
        let vocabURL = models.appendingPathComponent("kokoro-82m-coreml/ANE/vocab.json")
        let payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: lexicon))
        let vocab = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL)) as? [String: Any] ?? [:]
        let allowed = Set(vocab.keys)
        let join = { (m: [String: [String]]) in m.mapValues { $0.filter { allowed.contains($0) }.joined() } }
        return Inflections(lower: join(payload.lower), caseSensitive: join(payload.caseSensitive))
    }

    /// FluidAudio's direct lexicon lookup (KokoroAneEnglishPhonemizer.lookupMisakiWord).
    func lookup(_ word: String) -> String? {
        let n = Self.normalizeKey(word)
        let p = caseSensitive[word] ?? caseSensitive[n] ?? lower[word.lowercased()] ?? lower[n]
        guard let p, !p.isEmpty else { return nil }
        return p
    }

    static func normalizeKey(_ word: String) -> String {
        let allowed = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))
        return String(String.UnicodeScalarView(word.lowercased().unicodeScalars.filter { allowed.contains($0) }))
    }

    /// Phonemes for a word FluidAudio would otherwise hand to its G2P model, or nil. Only plain alphabetic words that
    /// miss the lexicon; all-caps words are left alone (FluidAudio spells initialisms), as are words with apostrophes
    /// or hyphens (FluidAudio's possessive and compound rules).
    func derive(_ word: String) -> String? {
        guard word.count > 1, word.allSatisfy({ $0.isLetter }), word != word.uppercased(), lookup(word) == nil else {
            return nil
        }
        // Misaki passes stress 0.5 for a capitalized word, and for -ing always (stem_ing's `0.5 if stress is None`).
        let capitalized = word != word.lowercased()
        if let stem = stemS(word) { return Self.s(capitalized ? Self.halfStress(stem) : stem) }
        if let stem = stemEd(word) { return Self.ed(capitalized ? Self.halfStress(stem) : stem) }
        if let stem = stemIng(word) { return Self.ing(Self.halfStress(stem)) }
        return nil
    }

    /// Entries for every word of `text` that `derive` resolves.
    func entries(for text: String) -> [String: String] {
        var out: [String: String] = [:]
        for token in text.split(whereSeparator: { !($0.isLetter || $0.isNumber || $0 == "'" || $0 == "-") }) {
            let w = String(token)
            if out[w] == nil, let p = derive(w) { out[w] = p }
        }
        return out
    }

    // MARK: - Misaki's rules

    func stemS(_ w: String) -> String? {
        let l = w.lowercased()
        if l.count > 2, l.hasSuffix("s"), !l.hasSuffix("ss"), let p = lookup(String(w.dropLast())) { return p }
        if l.count > 4, l.hasSuffix("es"), let p = lookup(String(w.dropLast(2))) { return p }
        if l.count > 4, l.hasSuffix("ies"), let p = lookup(String(w.dropLast(3)) + "y") { return p }
        return nil
    }

    func stemEd(_ w: String) -> String? {
        let l = w.lowercased()
        if l.hasSuffix("d"), !l.hasSuffix("dd"), let p = lookup(String(w.dropLast())) { return p }
        if l.hasSuffix("ed"), !l.hasSuffix("eed"), let p = lookup(String(w.dropLast(2))) { return p }
        return nil
    }

    func stemIng(_ w: String) -> String? {
        let l = w.lowercased()
        guard l.hasSuffix("ing") else { return nil }
        if let p = lookup(String(w.dropLast(3))) { return p }
        if let p = lookup(String(w.dropLast(3)) + "e") { return p }
        let c = Array(l)
        let doubled = c.count >= 5 && c[c.count - 4] == c[c.count - 5] && "bcdgklmnprstvxz".contains(c[c.count - 4])
        if doubled || l.hasSuffix("cking"), let p = lookup(String(w.dropLast(4))) { return p }
        return nil
    }

    static func s(_ stem: String) -> String {
        guard let last = stem.last else { return stem }
        if "ptkfθ".contains(last) { return stem + "s" }
        if "szʃʒʧʤ".contains(last) { return stem + "ᵻz" }
        return stem + "z"
    }

    static func ed(_ stem: String) -> String {
        let c = Array(stem)
        guard let last = c.last else { return stem }
        if "pkfθʃsʧ".contains(last) { return stem + "t" }
        if last == "d" { return stem + "ᵻd" }
        if last != "t" { return stem + "d" }
        if c.count < 2 { return stem + "ɪd" }
        if usTaus.contains(c[c.count - 2]) { return String(c.dropLast()) + "ɾᵻd" }
        return stem + "ᵻd"
    }

    static func ing(_ stem: String) -> String {
        let c = Array(stem)
        if c.count > 1, c.last == "t", usTaus.contains(c[c.count - 2]) { return String(c.dropLast()) + "ɾɪŋ" }
        return stem + "ɪŋ"
    }

    /// misaki apply_stress(ps, 0.5): a stem with no stress mark gets a secondary stress before its first vowel.
    static func halfStress(_ ps: String) -> String {
        guard !ps.contains(where: { stresses.contains($0) }), let i = ps.firstIndex(where: { vowels.contains($0) }) else {
            return ps
        }
        var r = ps
        r.insert("ˌ", at: i)
        return r
    }
}
