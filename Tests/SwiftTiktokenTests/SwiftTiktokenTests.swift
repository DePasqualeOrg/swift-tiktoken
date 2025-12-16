@testable import SwiftTiktoken
import XCTest

final class SwiftTiktokenTests: XCTestCase {
  // MARK: - Setup

  override class func setUp() {
    super.setUp()
    // Download vocabularies once for all tests
    let semaphore = DispatchSemaphore(value: 0)
    Task {
      await downloadVocabulariesIfNeeded()
      semaphore.signal()
    }
    semaphore.wait()
  }

  /// Download vocabularies if not already cached
  static func downloadVocabulariesIfNeeded() async {
    let vocabularies = [
      ("cl100k_base", "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"),
      ("r50k_base", "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"),
      ("p50k_base", "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"),
      ("o200k_base", "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"),
    ]

    // Use /tmp for sandboxed tests
    let cacheDir = URL(fileURLWithPath: "/tmp/tiktoken", isDirectory: true)

    // Set the custom cache directory for the loader
    EncodingLoader.customCacheDirectory = cacheDir

    // Check if already downloaded
    var needsDownload = false
    for (name, _) in vocabularies {
      let cacheFile = cacheDir.appendingPathComponent("\(name).tiktoken")
      if !FileManager.default.fileExists(atPath: cacheFile.path) {
        needsDownload = true
        break
      }
    }

    guard needsDownload else {
      print("✅ Vocabularies already cached")
      return
    }

    print("📥 Downloading vocabularies...")

    do {
      try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

      for (name, urlString) in vocabularies {
        let cacheFile = cacheDir.appendingPathComponent("\(name).tiktoken")

        // Skip if already exists
        if FileManager.default.fileExists(atPath: cacheFile.path) {
          print("✅ \(name) already cached")
          continue
        }

        print("⬇️  Downloading \(name)...")
        let url = URL(string: urlString)!
        let (data, _) = try await URLSession.shared.data(from: url)
        try data.write(to: cacheFile)
        print("✅ Downloaded \(name) (\(data.count) bytes)")
      }
    } catch {
      print("⚠️ Error downloading vocabularies: \(error)")
    }
  }

  // MARK: - Tests with Real Encodings

  func testCl100kBase() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    // Test known token values for cl100k_base
    XCTAssertEqual(encoder.encode(text: "hello world", allowedSpecial: []), [15339, 1917])
    XCTAssertEqual(encoder.encode(text: "", allowedSpecial: []), [])
    XCTAssertEqual(encoder.encode(text: " ", allowedSpecial: []), [220])

    // Test special tokens
    XCTAssertEqual(encoder.encode(text: "hello <|endoftext|>", allowedSpecial: ["<|endoftext|>"]), [15339, 220, 100_257])

    // Test regex patterns
    XCTAssertEqual(encoder.encode(text: "rer", allowedSpecial: []), [38149])
    XCTAssertEqual(encoder.encode(text: "'rer", allowedSpecial: []), [2351, 81])
    XCTAssertEqual(encoder.encode(text: "today\n ", allowedSpecial: []), [31213, 198, 220])

    // Test emoji
    XCTAssertEqual(encoder.encode(text: "👍", allowedSpecial: []), [9468, 239, 235])

    // Test roundtrip
    let text = "The quick brown fox jumps over the lazy dog."
    let tokens = encoder.encode(text: text, allowedSpecial: [])
    let decoded = try encoder.decode(tokens: tokens)
    XCTAssertEqual(decoded, text)
  }

  func testR50kBase() async throws {
    let encoder = try await CoreBPE.r50kBase()

    // Test known token values
    XCTAssertEqual(encoder.encode(text: "hello world", allowedSpecial: []), [31373, 995])

    // Test roundtrip
    let text = "Testing r50k_base encoding"
    let tokens = encoder.encode(text: text, allowedSpecial: [])
    let decoded = try encoder.decode(tokens: tokens)
    XCTAssertEqual(decoded, text)
  }

  func testP50kBase() async throws {
    let encoder = try await CoreBPE.p50kBase()

    // Test known token values
    XCTAssertEqual(encoder.encode(text: "hello world", allowedSpecial: []), [31373, 995])

    // Test roundtrip
    let text = "Testing p50k_base encoding"
    let tokens = encoder.encode(text: text, allowedSpecial: [])
    let decoded = try encoder.decode(tokens: tokens)
    XCTAssertEqual(decoded, text)
  }

  func testGPT2() async throws {
    // GPT-2 uses r50k_base encoding
    let encoder = try await CoreBPE.loadEncoding(named: "r50k_base")

    // Test known token values for GPT-2
    XCTAssertEqual(encoder.encode(text: "hello world", allowedSpecial: []), [31373, 995])
    XCTAssertEqual(encoder.encode(text: "hello <|endoftext|>", allowedSpecial: ["<|endoftext|>"]), [31373, 220, 50256])

    // Test repeated zeros
    XCTAssertEqual(encoder.encode(text: "0", allowedSpecial: []), [15])
    XCTAssertEqual(encoder.encode(text: "00", allowedSpecial: []), [405])
    XCTAssertEqual(encoder.encode(text: "000", allowedSpecial: []), [830])
    XCTAssertEqual(encoder.encode(text: "0000", allowedSpecial: []), [2388])
    XCTAssertEqual(encoder.encode(text: "00000", allowedSpecial: []), [20483])
    XCTAssertEqual(encoder.encode(text: "000000", allowedSpecial: []), [10535])
    XCTAssertEqual(encoder.encode(text: "0000000", allowedSpecial: []), [24598])
    XCTAssertEqual(encoder.encode(text: "00000000", allowedSpecial: []), [8269])
  }

  func testO200kBase() async throws {
    do {
      let encoder = try await CoreBPE.o200kBase()

      // Test basic functionality
      let text = "Testing o200k_base encoding"
      let tokens = encoder.encode(text: text, allowedSpecial: [])
      XCTAssertFalse(tokens.isEmpty)

      // Test roundtrip
      let decoded = try encoder.decode(tokens: tokens)
      XCTAssertEqual(decoded, text)
    } catch {
      // o200k_base might not be available in all versions
      throw XCTSkip("o200k_base not available: \(error)")
    }
  }

  // MARK: - Cross-Encoding Comparison Tests

  func testEncodingComparison() async throws {
    let text = "The quick brown fox jumps over the lazy dog."

    let cl100k = try await CoreBPE.cl100kBase()
    let r50k = try await CoreBPE.r50kBase()
    let p50k = try await CoreBPE.p50kBase()

    let cl100kTokens = cl100k.encode(text: text, allowedSpecial: [])
    let r50kTokens = r50k.encode(text: text, allowedSpecial: [])
    let p50kTokens = p50k.encode(text: text, allowedSpecial: [])

    // r50k and p50k should produce the same tokens for basic text
    XCTAssertEqual(r50kTokens, p50kTokens)

    // cl100k should be different (more efficient)
    XCTAssertNotEqual(cl100kTokens, r50kTokens)

    // cl100k should generally use fewer tokens
    XCTAssertLessThanOrEqual(cl100kTokens.count, r50kTokens.count)

    print("Token counts for '\(text)':")
    print("  cl100k_base: \(cl100kTokens.count) tokens")
    print("  r50k_base:   \(r50kTokens.count) tokens")
    print("  p50k_base:   \(p50kTokens.count) tokens")
  }

  // MARK: - Special Token Tests

  func testSpecialTokens() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let text = "Hello <|endoftext|> World"

    // Without allowing special tokens (should encode the literal text)
    let withoutSpecial = encoder.encodeOrdinary(text: text)

    // With special tokens allowed
    let withSpecial = encoder.encode(text: text, allowedSpecial: ["<|endoftext|>"])

    // These should be different
    XCTAssertNotEqual(withoutSpecial, withSpecial)

    // The version with special tokens should contain token 100257 (endoftext)
    XCTAssertTrue(withSpecial.contains(100_257))
    XCTAssertFalse(withoutSpecial.contains(100_257))
  }

  // MARK: - Catastrophic Repetition Test

  func testCatastrophicallyRepetitive() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let testChars = ["^", "0", "a", "'s", " ", "\n"]

    for char in testChars {
      // Test with large repetition
      let bigValue = String(repeating: char, count: 10000)
      let tokens = encoder.encode(text: bigValue, allowedSpecial: [])
      let decoded = try encoder.decode(tokens: tokens)
      XCTAssertEqual(decoded, bigValue, "Failed for repeated: \(char)")

      // Test with space prefix
      let withPrefix = " " + bigValue
      let tokensPrefix = encoder.encode(text: withPrefix, allowedSpecial: [])
      let decodedPrefix = try encoder.decode(tokens: tokensPrefix)
      XCTAssertEqual(decodedPrefix, withPrefix, "Failed with prefix for: \(char)")

      // Test with newline suffix
      let withSuffix = bigValue + "\n"
      let tokensSuffix = encoder.encode(text: withSuffix, allowedSpecial: [])
      let decodedSuffix = try encoder.decode(tokens: tokensSuffix)
      XCTAssertEqual(decodedSuffix, withSuffix, "Failed with suffix for: \(char)")
    }
  }

  // MARK: - International Text Tests

  func testInternationalText() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let testCases = [
      "请考试我的软件！12345",
      "こんにちは世界",
      "مرحبا بالعالم",
      "Здравствуй, мир",
      "Bonjour le monde",
      "🚀🌍🎉",
    ]

    for text in testCases {
      let tokens = encoder.encode(text: text, allowedSpecial: [])
      let decoded = try encoder.decode(tokens: tokens)
      XCTAssertEqual(decoded, text, "Roundtrip failed for: \(text)")
    }
  }

  // MARK: - Edge Cases

  func testEdgeCases() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    // Empty string
    XCTAssertEqual(encoder.encode(text: "", allowedSpecial: []), [])

    // Single space
    XCTAssertEqual(encoder.encode(text: " ", allowedSpecial: []), [220])

    // Newline
    XCTAssertEqual(encoder.encode(text: "\n", allowedSpecial: []), [198])

    // Tab
    let tabTokens = encoder.encode(text: "\t", allowedSpecial: [])
    XCTAssertFalse(tabTokens.isEmpty)

    // Mixed whitespace
    let whitespace = " \t\n\r"
    let wsTokens = encoder.encode(text: whitespace, allowedSpecial: [])
    let wsDecoded = try encoder.decode(tokens: wsTokens)
    XCTAssertEqual(wsDecoded, whitespace)
  }

  // MARK: - Performance Tests

  func testEncodingPerformance() async throws {
    let encoder = try await CoreBPE.cl100kBase()
    let text = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 1000)

    measure {
      _ = encoder.encode(text: text, allowedSpecial: [])
    }
  }

  func testDecodingPerformance() async throws {
    let encoder = try await CoreBPE.cl100kBase()
    let text = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 1000)
    let tokens = encoder.encode(text: text, allowedSpecial: [])

    measure {
      _ = try? encoder.decode(tokens: tokens)
    }
  }

  // MARK: - Thread Safety

  func testConcurrentEncoding() async throws {
    let encoder = try await CoreBPE.cl100kBase()
    let expectation = XCTestExpectation(description: "Concurrent encoding")
    let iterations = 100
    expectation.expectedFulfillmentCount = iterations

    DispatchQueue.concurrentPerform(iterations: iterations) { i in
      let text = "Concurrent test \(i)"
      let tokens = encoder.encode(text: text, allowedSpecial: [])
      if let decodedText = try? encoder.decode(tokens: tokens) {
        XCTAssertEqual(decodedText, text)
      }
      expectation.fulfill()
    }

    await fulfillment(of: [expectation], timeout: 10.0)
  }

  // MARK: - BPE Algorithm Tests

  func testBytePairMergeBasic() {
    // Create simple ranks
    var ranks: [ArraySlice<UInt8>: Rank] = [:]
    ranks[ArraySlice([UInt8(ascii: "a"), UInt8(ascii: "b")])] = 0
    ranks[ArraySlice([UInt8(ascii: "c"), UInt8(ascii: "d")])] = 1

    let piece: [UInt8] = [UInt8(ascii: "a"), UInt8(ascii: "b"), UInt8(ascii: "c"), UInt8(ascii: "d")]

    let result = bytePairSplit(piece: ArraySlice(piece), ranks: ranks)
    XCTAssertEqual(result.count, 2)
    XCTAssertEqual(Array(result[0]), [UInt8(ascii: "a"), UInt8(ascii: "b")])
    XCTAssertEqual(Array(result[1]), [UInt8(ascii: "c"), UInt8(ascii: "d")])
  }

  func testBytePairMergeRepeated() {
    var ranks: [ArraySlice<UInt8>: Rank] = [:]
    ranks[ArraySlice([UInt8(ascii: "a"), UInt8(ascii: "b")])] = 0
    ranks[ArraySlice([UInt8(ascii: "c"), UInt8(ascii: "d")])] = 1

    let piece: [UInt8] = [UInt8(ascii: "a"), UInt8(ascii: "b"), UInt8(ascii: "a"), UInt8(ascii: "b")]

    let result = bytePairSplit(piece: ArraySlice(piece), ranks: ranks)
    XCTAssertEqual(result.count, 2)
    XCTAssertEqual(Array(result[0]), [UInt8(ascii: "a"), UInt8(ascii: "b")])
    XCTAssertEqual(Array(result[1]), [UInt8(ascii: "a"), UInt8(ascii: "b")])
  }

  // MARK: - Decode Error Tests

  func testDecodeInvalidToken() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    // Try to decode with an invalid token
    XCTAssertThrowsError(try encoder.decodeBytes(tokens: [999_999_999])) { error in
      if case let TiktokenError.decodeKeyError(token) = error {
        XCTAssertEqual(token, 999_999_999)
      } else {
        XCTFail("Expected decodeKeyError")
      }
    }
  }

  // MARK: - Single Token Operations Tests

  func testEncodeSingleToken() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    // Test encoding a single known token
    let helloBytes = Array("hello".utf8)
    let token = try encoder.encodeSingleToken(piece: helloBytes)
    XCTAssertEqual(token, 15339)

    // Test that multi-token pieces throw
    let longText = Array("hello world".utf8)
    XCTAssertThrowsError(try encoder.encodeSingleToken(piece: longText))
  }

  func testEncodeSinglePiece() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    // Test single token piece
    let helloBytes = Array("hello".utf8)
    let tokens = encoder.encodeSinglePiece(piece: helloBytes)
    XCTAssertEqual(tokens, [15339])

    // Test multi-token piece
    let longText = Array("hello world".utf8)
    let multiTokens = encoder.encodeSinglePiece(piece: longText)
    XCTAssertEqual(multiTokens, [15339, 1917])
  }

  func testDecodeSingleTokenBytes() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    // Test decoding known token
    let bytes = try encoder.decodeSingleTokenBytes(token: 15339)
    XCTAssertEqual(String(bytes: bytes, encoding: .utf8), "hello")

    // Test invalid token throws
    XCTAssertThrowsError(try encoder.decodeSingleTokenBytes(token: 999_999_999))
  }

  func testTokenByteValues() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let allTokens = encoder.tokenByteValues()
    // cl100k_base has ~100k tokens
    XCTAssertGreaterThan(allTokens.count, 90000)
    XCTAssertLessThan(allTokens.count, 110_000)

    // Verify sorted order
    for i in 1 ..< min(100, allTokens.count) {
      XCTAssertTrue(allTokens[i - 1].lexicographicallyPrecedes(allTokens[i]) ||
        allTokens[i - 1] == allTokens[i])
    }
  }

  func testEncodeBytes() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    // Test valid UTF-8
    let validBytes = Array("hello world".utf8)
    let tokens = encoder.encodeBytes(validBytes)
    XCTAssertEqual(tokens, [15339, 1917])

    // Test roundtrip
    let decoded = try encoder.decode(tokens: tokens)
    XCTAssertEqual(decoded, "hello world")
  }

  // MARK: - Encode With Special Tokens Tests

  func testEncodeWithSpecialTokens() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let text = "Hello<|endoftext|>World<|fim_prefix|>End"

    // All special tokens allowed
    let tokens = encoder.encodeWithSpecialTokens(text: text)

    // Should contain the special token values
    XCTAssertTrue(tokens.contains(100_257)) // endoftext
    XCTAssertTrue(tokens.contains(100_258)) // fim_prefix

    // Roundtrip should work
    let decoded = try encoder.decode(tokens: tokens)
    XCTAssertEqual(decoded, text)
  }

  // MARK: - Decode With Offsets Tests

  func testDecodeWithOffsetsBasic() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let prompt = "hello world"
    let tokens = encoder.encode(text: prompt, allowedSpecial: [])
    let (text, offsets) = try encoder.decodeWithOffsets(tokens: tokens)

    XCTAssertEqual(text, prompt)
    XCTAssertEqual(offsets, [0, 5])
  }

  func testDecodeWithOffsetsSpecialTokens() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let prompt = "hello world<|endoftext|> green cow"
    let tokens = encoder.encodeWithSpecialTokens(text: prompt)
    let (text, offsets) = try encoder.decodeWithOffsets(tokens: tokens)

    XCTAssertEqual(text, prompt)
    XCTAssertEqual(offsets, [0, 5, 11, 24, 30])
  }

  func testDecodeWithOffsetsChinese() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let prompt = "我非常渴望与人工智能一起工作"
    let tokens = encoder.encode(text: prompt, allowedSpecial: [])
    let (text, offsets) = try encoder.decodeWithOffsets(tokens: tokens)

    XCTAssertEqual(text, prompt)
    XCTAssertEqual(offsets.count, tokens.count)
    XCTAssertEqual(offsets.first, 0)
  }

  // MARK: - Batch Encoding Tests

  func testBatchEncode() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let texts = ["hello world", "goodbye world", "test"]
    let batch = await encoder.encodeBatch(texts)

    XCTAssertEqual(batch.count, texts.count)
    XCTAssertEqual(batch[0], encoder.encode(text: texts[0], allowedSpecial: []))
    XCTAssertEqual(batch[1], encoder.encode(text: texts[1], allowedSpecial: []))
    XCTAssertEqual(batch[2], encoder.encode(text: texts[2], allowedSpecial: []))
  }

  func testBatchEncodeOrdinary() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let texts = ["hello world", "goodbye world", "test"]
    let batch = await encoder.encodeOrdinaryBatch(texts)

    XCTAssertEqual(batch.count, texts.count)
    XCTAssertEqual(batch[0], encoder.encodeOrdinary(text: texts[0]))
    XCTAssertEqual(batch[1], encoder.encodeOrdinary(text: texts[1]))
    XCTAssertEqual(batch[2], encoder.encodeOrdinary(text: texts[2]))
  }

  func testBatchDecode() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let texts = ["hello world", "goodbye world", "test"]
    let batch = await encoder.encodeBatch(texts)
    let decoded = try await encoder.decodeBatch(batch)

    XCTAssertEqual(decoded, texts)
  }

  func testBatchRoundtrip() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let texts = [
      "The quick brown fox",
      "jumps over the lazy dog",
      "Hello, 世界!",
      "🚀 to the moon",
    ]

    let encoded = await encoder.encodeBatch(texts)
    let decoded = try await encoder.decodeBatch(encoded)

    XCTAssertEqual(decoded, texts)
  }

  func testBatchSyncEncode() async throws {
    let encoder = try await CoreBPE.cl100kBase()

    let texts = ["hello world", "goodbye world", "test"]
    let batch = encoder.encodeBatchSync(texts)

    XCTAssertEqual(batch.count, texts.count)
    XCTAssertEqual(batch[0], encoder.encode(text: texts[0], allowedSpecial: []))
    XCTAssertEqual(batch[1], encoder.encode(text: texts[1], allowedSpecial: []))
    XCTAssertEqual(batch[2], encoder.encode(text: texts[2], allowedSpecial: []))
  }

  // MARK: - Model Encoding Tests

  func testEncodingForModel() async throws {
    // Test GPT-4
    let gpt4 = try await CoreBPE.forModel("gpt-4")
    XCTAssertEqual(gpt4.encode(text: "hello world", allowedSpecial: []), [15339, 1917])

    // Test GPT-3.5
    let gpt35 = try await CoreBPE.forModel("gpt-3.5-turbo")
    XCTAssertEqual(gpt35.encode(text: "hello world", allowedSpecial: []), [15339, 1917])

    // Test GPT-4o
    let gpt4o = try await CoreBPE.forModel("gpt-4o")
    XCTAssertNotNil(gpt4o)

    // Test GPT-2
    let gpt2 = try await CoreBPE.forModel("gpt2")
    XCTAssertEqual(gpt2.encode(text: "hello world", allowedSpecial: []), [31373, 995])

    // Test text-davinci-003
    let davinci003 = try await CoreBPE.forModel("text-davinci-003")
    XCTAssertNotNil(davinci003)
  }

  func testEncodingForModelPrefixes() async throws {
    // Test versioned models
    let gpt4_0314 = try await CoreBPE.forModel("gpt-4-0314")
    XCTAssertEqual(gpt4_0314.encode(text: "hello world", allowedSpecial: []), [15339, 1917])

    let gpt35turbo_0301 = try await CoreBPE.forModel("gpt-3.5-turbo-0301")
    XCTAssertEqual(gpt35turbo_0301.encode(text: "hello world", allowedSpecial: []), [15339, 1917])
  }

  func testEncodingForModelUnknown() async throws {
    do {
      _ = try await CoreBPE.forModel("unknown-model-xyz")
      XCTFail("Should throw for unknown model")
    } catch {
      // Expected
    }
  }

  func testModelEncodingMapping() {
    // Test exact matches
    XCTAssertEqual(ModelEncoding.encodingName(forModel: "gpt-4"), "cl100k_base")
    XCTAssertEqual(ModelEncoding.encodingName(forModel: "gpt-4o"), "o200k_base")
    XCTAssertEqual(ModelEncoding.encodingName(forModel: "gpt2"), "gpt2")
    XCTAssertEqual(ModelEncoding.encodingName(forModel: "text-davinci-003"), "p50k_base")

    // Test prefix matches
    XCTAssertEqual(ModelEncoding.encodingName(forModel: "gpt-4-0314"), "cl100k_base")
    XCTAssertEqual(ModelEncoding.encodingName(forModel: "gpt-4o-2024-05-13"), "o200k_base")
    XCTAssertEqual(ModelEncoding.encodingName(forModel: "gpt-3.5-turbo-0301"), "cl100k_base")

    // Test unknown
    XCTAssertNil(ModelEncoding.encodingName(forModel: "unknown-model"))
  }
}
