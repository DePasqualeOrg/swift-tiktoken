// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "SwiftTiktoken",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
  ],
  products: [
    .library(
      name: "SwiftTiktoken",
      targets: ["SwiftTiktoken"],
    ),
  ],
  targets: [
    .target(
      name: "SwiftTiktoken",
      path: "Sources/SwiftTiktoken"
    ),
    .executableTarget(
      name: "Benchmark",
      dependencies: ["SwiftTiktoken"],
      path: "Sources/Benchmark"
    ),
    .testTarget(
      name: "SwiftTiktokenTests",
      dependencies: ["SwiftTiktoken"],
      path: "Tests/SwiftTiktokenTests"
    ),
  ],
)
