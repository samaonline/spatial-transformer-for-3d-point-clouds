# datasets

### ShapeNet Part
1. Download and uncompress the file from the [PointNet++ repo](https://github.com/charlesq34/pointnet2).
 The link is located under the [Object Part Segmentation](https://github.com/charlesq34/pointnet2/blob/master/README.md#object-part-segmentation)
section. 
2. Convert to PLY files: 
    ```bash
    # this step can take 10~15 minutes
    python shapenet_prepare.py <PATH_TO_DOWNLOADED_FOLDER>
    ``` 
    Check that folder `data/shapenet_ericyi_ply` is now generated filled with PLY files. 


### References

ShapeNet point clouds and annotations were originally collected by the authors of: 
- L. Yi, et al. A Scalable Active Framework for Region Annotation in 3D Shape Collections. SIGGRAPH Asia, 2016.

Normal directions of the point clouds are provided by the authors of: 
- C. R. Qi, et al. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NIPS, 2017. 
