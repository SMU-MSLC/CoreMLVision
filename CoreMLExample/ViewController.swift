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

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    //MARK: UI View Elements
    @IBOutlet weak var mainImageView: UIImageView!
    @IBOutlet weak var classifierLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    //MARK: ML Model Load
    // Load an image classifier and encapsulate in the Vision model class
    lazy var model: VNCoreMLModel? = {
        guard let tmpModel = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
            return nil
        }
       return tmpModel
    }()
    
    // select some different classification models!
    @IBAction func modelSelectChanged(_ sender: UISegmentedControl) {
        switch sender.selectedSegmentIndex {
        case 0:
            guard let tmpModel = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
                return
            }
            model = tmpModel
        case 1:
            guard let tmpModel = try? VNCoreMLModel(for: SqueezeNet().model) else {
                return
            }
            model = tmpModel
        case 2:
            guard let tmpModel = try? VNCoreMLModel(for: Resnet50().model) else {
                return
            }
            model = tmpModel
        default:
            return
        }
        // update UI if we changed and an image exists
        if let image = self.mainImageView.image {
            classifyImage(image: image)
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
        
        classifyImage(image: image)
        mainImageView.image = image
    }
    
    //MARK: Custom Classification Methods
    // use vision API to classify image
    func classifyImage(image:UIImage){
        let request = VNCoreMLRequest(model: self.model!, completionHandler: resultsMethod)
        let handler = VNImageRequestHandler(cgImage: image.cgImage!, options: [:])
        do{
            try handler.perform([request])
        }catch _{
            self.classifierLabel.text = "Error, could not classify"
        }
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
