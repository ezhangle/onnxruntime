// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
class IExecutionProvider;

struct IExecutionProviderFactory {
  virtual std::unique_ptr<IExecutionProvider> CreateProvider() = 0;
};
}  // namespace onnxruntime
