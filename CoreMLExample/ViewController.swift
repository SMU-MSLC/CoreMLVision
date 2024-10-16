//
//  ViewController.swift
//  CoreMLExample
//
//  Created by Eric Larson on 9/5/17.
//  Copyright © 2017 Eric Larson. All rights reserved.
//

import UIKit
import CoreML
import Vision
import CoreImage

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    //MARK: UI View Elements
    @IBOutlet weak var mainImageView: UIImageView!
    @IBOutlet weak var classifierLabel: UILabel!
    
    
    // Image classification models (from CoreML)
    // this model can classify regions and common places (bathroom, living room, etc.)
    lazy var googLeNet:GoogLeNetPlaces = {
        do{
            let config = MLModelConfiguration()
            return try GoogLeNetPlaces(configuration: config)
        }catch{
            print(error)
            fatalError("Could not load GoogLeNet")
        }
    }()
    
    // this model can classify objects, using a very tiny model
    lazy var squeezeNet:SqueezeNet = {
        do{
            let config = MLModelConfiguration()
            return try SqueezeNet(configuration: config)
        }catch{
            print(error)
            fatalError("Could not load SqueezeNet")
        }
    }()
    
    // this model can classify objects, using a larger model
    lazy var resNet:Resnet50 = {
        do{
            let config = MLModelConfiguration()
            return try Resnet50(configuration: config)
        }catch{
            print(error)
            fatalError("Could not load Resnet50")
        }
    }()
    
    // a state variable for tracking if the image has already been pre-processed
    var needProcessing = true

    
    //MARK: ML Model Load
    // Load an image classifier and encapsulate in the Vision model class
    lazy var model: VNCoreMLModel? = {
        guard let tmpModel = try? VNCoreMLModel(for: googLeNet.model) else {
            return nil
        }
       return tmpModel
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    // select some different classification models!
    @IBAction func modelSelectChanged(_ sender: UISegmentedControl) {
        switch sender.selectedSegmentIndex {
        case 0:
            guard let tmpModel = try? VNCoreMLModel(for: googLeNet.model) else {
                return
            }
            model = tmpModel
        case 1:
            guard let tmpModel = try? VNCoreMLModel(for: squeezeNet.model) else {
                return
            }
            model = tmpModel
        case 2:
            guard let tmpModel = try? VNCoreMLModel(for: resNet.model) else {
                return
            }
            model = tmpModel
        default:
            return
        }
        // update UI if we changed and an image exists
        if let image = self.mainImageView.image {
            _ = classifyImage(image: image)
        }
    }
    
    
    //MARK: Camera View Presentation
    @IBAction func takePicture(_ sender: UIButton) {
        
        if !UIImagePickerController.isSourceTypeAvailable(.camera) {
            return
        }
        
        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = (self as UIImagePickerControllerDelegate & UINavigationControllerDelegate)
        cameraPicker.sourceType = .camera
        cameraPicker.allowsEditing = false
        present(cameraPicker, animated: true)
    }
}


//MARK: Extension Class for Classification
extension ViewController: UIImagePickerControllerDelegate {
    
    //MARK: Camera View Callbacks
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true)
        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        }
        
        // perform this on a background queue
        DispatchQueue.global().async {
            self.needProcessing = true
            let newImage = self.classifyImage(image: image)
            
            // use update on new queue
            DispatchQueue.main.async{
                self.mainImageView.image = newImage
            }
        }
        
    }
    
    //MARK: Custom Classification Methods
    // use vision API to classify image
    func classifyImage(image:UIImage) -> (UIImage){
        // Filters: -crop image so it isn't squashed
        //          -increase contrast
        //          -add some blurring/noise filters
        
        // code to convert CI to CG Image from here: http://wiki.hawkguide.com/wiki/Swift:_Convert_between_CGImage,_CIImage_and_UIImage
        func convertCIImageToCGImage(inputImage: CIImage) -> CGImage! {
            let context = CIContext(options: nil)
            if let cgImage = context.createCGImage(inputImage, from: inputImage.extent){
                return cgImage
            }
            return nil
        }
        
        var cgImage: CGImage? = nil
        
        if self.needProcessing {
            // pre-processing steps:
            // 1. crop to mult*224 by mult*224
            // 2. scale to mult (so image is roughly 224 by 224)
            // 3. enhance contrast
        
            // try to apply a cropping filter
            var ciImage = CIImage(cgImage: image.cgImage!)
            let filter = CIFilter(name:"CICrop")
            let mult:Float = 12
            filter?.setValue(CIVector(x: 1, y: 1, z: 1+224*CGFloat(mult), w: 1+224*CGFloat(mult)), forKey: "inputRectangle")
            filter?.setValue(ciImage, forKey: "inputImage")
            
            ciImage = (filter?.outputImage)!
            
            // apply filter for scaling image by factor of 1/3
            // as the image is expected to be 224x224 for these models
            let filter2 = CIFilter(name:"CILanczosScaleTransform")
            filter2?.setValue(1.0/mult, forKey: "inputScale")
            filter2?.setValue(ciImage, forKey: "inputImage")
            ciImage = (filter2?.outputImage)!
            
            cgImage = convertCIImageToCGImage(inputImage: ciImage)
            
            // enhance contrast of image
            let filter3 = CIFilter(name:"CIColorControls")
            filter3?.setValue(1.0, forKey: "inputContrast")
            filter3?.setValue(ciImage, forKey: "inputImage")
            ciImage = (filter3?.outputImage)!
            
            cgImage = convertCIImageToCGImage(inputImage: ciImage)
            self.needProcessing = false
        }
        else{
            // if no processing needed, just set image
            cgImage = image.cgImage
        }
        
        
        // generate request for vision and ML model
        let request = VNCoreMLRequest(model: self.model!,
                                      completionHandler: resultsMethod)
        
        // add data to vision request handler
        let handler = VNImageRequestHandler(cgImage: cgImage!,
                                            options: [:])
        
        // now perform classification
        do{
            try handler.perform([request])
            // completion handler gets request, asynchronously
        }catch _{
            self.classifierLabel.text = "Error, could not classify"
        }
        
        // return the UIIMage for display (so we see what we tried to classify
        return UIImage(cgImage: cgImage!, scale: image.scale, orientation: image.imageOrientation)
    }
    
    //interpret results from vision query
    func resultsMethod(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNClassificationObservation]
            else {
                fatalError("Could not cast request as classification object")
        }
        
        // Add in results display...
        print("---------------")
        for result in results {
            if(result.confidence > 0.05){
                print(result.identifier,result.confidence)
            }
        }
        
        DispatchQueue.main.async{
            self.classifierLabel.text = "This might be a \(results[0].identifier) \n conf:\(results[0].confidence)"
            
        }
        
    }
    

}
