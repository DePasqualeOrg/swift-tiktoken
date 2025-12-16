import Foundation

/// Errors that can occur during tiktoken operations
public enum TiktokenError: Error, Equatable, Hashable, LocalizedError {
  /// Error during regex compilation or matching
  case regexError(message: String)
  /// Error during token decoding
  case decodeError(message: String)
  /// Error when a token cannot be found in the decoder
  case decodeKeyError(token: Rank)
  /// Error during encoding
  case encodeError(message: String)

  public var errorDescription: String? {
    switch self {
      case let .regexError(message):
        "Regex error: \(message)"
      case let .decodeError(message):
        "Decode error: \(message)"
      case let .decodeKeyError(token):
        "Invalid token for decoding: \(token)"
      case let .encodeError(message):
        "Encode error: \(message)"
    }
  }
}
