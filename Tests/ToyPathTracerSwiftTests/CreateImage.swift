import CoreImage
import ToyPathTracerSwift

func createImage(_ pixelBufferReader: PixelBufferReader) -> CIImage {
  /// Create an Image from a BackBuffer.
  let sizeOfFloat = 4
  let pixelStride = 3
  let w = pixelBufferReader.w
  let h = pixelBufferReader.h
  let bytesPerRow = sizeOfFloat * pixelStride * w
  var data = Data(count: bytesPerRow * h)
  data.withUnsafeMutableBytes { (buffer: UnsafeMutableRawBufferPointer) in
    let ptr = buffer.bindMemory(to: Float.self)
    let componentsPerScanline = pixelStride * w
    for y in 0..<h {
      // Flip Y coordinate.
      var outputIndex = (h - y - 1) * componentsPerScanline
      for i in 0..<w {
        let pixel = pixelBufferReader.getPixel(x: i, y: y)
        ptr[outputIndex] = pixel.x
        outputIndex += 1
        ptr[outputIndex] = pixel.y
        outputIndex += 1
        ptr[outputIndex] = pixel.z
        outputIndex += 1
        
      }
    }
  }
  let convertedData: NSData = data as NSData
  let dataProvider = CGDataProvider(data: convertedData)!
  let bitmapInfo: CGBitmapInfo = [
    .byteOrder32Little,
    .floatComponents,
    CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
  ]
  
  let colorSpace = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!
  let cgImage = CGImage(width: w,
                        height: h,
                        bitsPerComponent: 32,
                        bitsPerPixel: 32*pixelStride,
                        bytesPerRow: bytesPerRow,
                        space: colorSpace,
                        bitmapInfo: bitmapInfo,
                        provider: dataProvider,
                        decode: nil,
                        shouldInterpolate: false,
                        intent: .defaultIntent)!
  let ciImage = CIImage(cgImage:cgImage)
  return ciImage
}
