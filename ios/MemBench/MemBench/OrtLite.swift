import Foundation
import onnxruntime

/// Minimal Swift wrapper over the ONNX Runtime C API, enough to run a model with int64/float
/// inputs and one float output. It uses the same onnxruntime.framework as sherpa-onnx, so the app
/// carries a single runtime.
enum OrtError: Error { case failed(String) }

final class OrtLite {
    private static let api: OrtApi = OrtGetApiBase()!.pointee.GetApi(UInt32(ORT_API_VERSION))!.pointee

    private var env: OpaquePointer?
    private var options: OpaquePointer?
    private var session: OpaquePointer?
    private var memory: OpaquePointer?

    private static func check(_ status: OpaquePointer?) throws {
        guard let status else { return }
        let message = api.GetErrorMessage(status).map { String(cString: $0) } ?? "unknown ONNX Runtime error"
        api.ReleaseStatus(status)
        throw OrtError.failed(message)
    }

    init(modelPath: String, threads: Int32 = 2) throws {
        let api = Self.api
        try Self.check(api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ortlite", &env))
        try Self.check(api.CreateSessionOptions(&options))
        try Self.check(api.SetIntraOpNumThreads(options, threads))
        try Self.check(api.CreateSession(env, modelPath, options, &session))
        try Self.check(api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory))
    }

    deinit {
        let api = Self.api
        if let memory { api.ReleaseMemoryInfo(memory) }
        if let session { api.ReleaseSession(session) }
        if let options { api.ReleaseSessionOptions(options) }
        if let env { api.ReleaseEnv(env) }
    }

    enum Input {
        case int64([Int64], shape: [Int64])
        case float([Float], shape: [Int64])
    }

    /// Runs the model and returns the first output as floats.
    func run(_ inputs: [(name: String, value: Input)], output: String) throws -> [Float] {
        let api = Self.api
        var values: [OpaquePointer?] = []
        defer { for v in values { if let v { api.ReleaseValue(v) } } }

        // Tensor data must outlive the call: keep it in arrays that stay alive until after Run.
        var int64Storage: [[Int64]] = []
        var floatStorage: [[Float]] = []
        for (_, input) in inputs {
            var value: OpaquePointer?
            switch input {
            case let .int64(data, shape):
                int64Storage.append(data)
                try int64Storage[int64Storage.count - 1].withUnsafeMutableBytes { raw in
                    try shape.withUnsafeBufferPointer { dims in
                        try Self.check(api.CreateTensorWithDataAsOrtValue(
                            memory, raw.baseAddress, raw.count, dims.baseAddress, dims.count,
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &value))
                    }
                }
            case let .float(data, shape):
                floatStorage.append(data)
                try floatStorage[floatStorage.count - 1].withUnsafeMutableBytes { raw in
                    try shape.withUnsafeBufferPointer { dims in
                        try Self.check(api.CreateTensorWithDataAsOrtValue(
                            memory, raw.baseAddress, raw.count, dims.baseAddress, dims.count,
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value))
                    }
                }
            }
            values.append(value)
        }

        let inputNames = inputs.map { strdup($0.name) }
        let outputName = strdup(output)
        defer { inputNames.forEach { free($0) }; free(outputName) }
        let namePointers = inputNames.map { UnsafePointer<CChar>($0) }
        var outPointer: [UnsafePointer<CChar>?] = [UnsafePointer(outputName)]
        var result: OpaquePointer?
        try Self.check(api.Run(session, nil, namePointers, values, values.count,
                               &outPointer, 1, &result))
        defer { if let result { api.ReleaseValue(result) } }

        var info: OpaquePointer?
        try Self.check(api.GetTensorTypeAndShape(result, &info))
        defer { if let info { api.ReleaseTensorTypeAndShapeInfo(info) } }
        var count = 0
        try Self.check(api.GetTensorShapeElementCount(info, &count))
        var raw: UnsafeMutableRawPointer?
        try Self.check(api.GetTensorMutableData(result, &raw))
        return Array(UnsafeBufferPointer(start: raw!.assumingMemoryBound(to: Float.self), count: count))
    }
}
