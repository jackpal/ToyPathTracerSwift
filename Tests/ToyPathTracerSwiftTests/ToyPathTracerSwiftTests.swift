import Testing
import ToyPathTracerSwift
import CoreImage

@Test
func testExample() {
        let point = float3(1,2,3)
    #expect(point.x == 1)
}

func writeFile(image: CIImage, path: String) throws {
  let lossyCompressionQuality = 0.76

  let options = [kCGImageDestinationLossyCompressionQuality
                 as CIImageRepresentationOption : lossyCompressionQuality]
  let context = CIContext()
  let colorSpace = image.colorSpace!
  let filename = URL(fileURLWithPath: path)
  let pathExtension = filename.pathExtension
  switch pathExtension.lowercased() {
  case "heic","heif":
    try context.writeHEIF10Representation(of: image.settingAlphaOne(in: image.extent),
                                        to: filename,
                                        colorSpace: colorSpace,
                                        options: options)
  case "jpeg", "jpg":
    try context.writeJPEGRepresentation(of: image,
                                        to: filename,
                                        colorSpace: colorSpace,
                                        options: options)

  case "png":
    try context.writePNGRepresentation(of: image,
                                       to: filename,
                                       format: .RGBA16,
                                       colorSpace: colorSpace,
                                       options: options)
  case "tif", "tiff":
    try context.writeTIFFRepresentation(of: image,
                                       to: filename,
                                       format: .RGBA16,
                                       colorSpace: colorSpace,
                                       options: options)
  default:
    print("Unknown file extension: \(pathExtension)")
  }
}

@Test("Trace a frame")
func timeTraceFrame() throws {
  let clock = ContinuousClock()

  let result = try clock.measure {
    let buffer = trace(width: 196, height: 100, frames: 4, threaded: true)
    let image = createImage(buffer)
    try writeFile(image:image, path:"/tmp/test.heif")
  }
  print("Benchmark: \(result)")
}
