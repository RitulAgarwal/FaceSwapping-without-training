import cv2 
import dlib
import numpy as np 

def index_from_array(nparray):
    index=None
    for n in nparray[0]:
        index = n
        break
    return index


#1.dlib used for detecting and predicting face iamge 
frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")
#print(type(frontal_face_detector),type(frontal_face_predictor))

#2.cv2 used for converting to graysacle the source and dest images
source_image = cv2.imread("images/2.png")
source_image_grayscale = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("SOURCE_IMAGE", source_image_grayscale)
destination_image = cv2.imread("images/1.png")
destination_image_grayscale = cv2.cvtColor(destination_image,cv2.COLOR_BGR2GRAY)
cv2.imshow("destination_IMAGE", destination_image_grayscale)

#3.create zeros array canvas fro src and dst imag
source_image_canvas = np.zeros_like(source_image_grayscale)
#print(source_image_canvas,type(source_image_canvas),source_image_canvas.shape)
#zeros obtains memory from the operating system so that the OS zeroes it when it is first used
#zeros_like on the other hand fills the alloced memory with zeros by itself
height,width ,no_of_channels = destination_image.shape
#print(height,width,no_of_channels)
#creating zeroes array cnavas with sixe of dst image
destination_image_canvas = np.zeros((height,width,no_of_channels),np.uint8)
#print(destination_image_canvas,type(destination_image_canvas),destination_image_canvas.shape)
#for source image---find faces in src image ---reutrns np array containing a histogram of pixels in image 
source_faces = frontal_face_detector(source_image_grayscale)
#print(source_faces,type(source_faces))

#lloop thrugh face,landmark points
#lp are 68 crucial points for fracial recog no check of oclour but only these 68 points 
#4.LOOP through all lfaces found in the source image
for source_face in source_faces:
    #print(source_face,np.shape(source_face))
    
    source_face_landmarks = frontal_face_predictor(source_image_grayscale,source_face)
    source_face_landmark_points = []
    #looping through all these 68 landmark points and add htme to the tuple 
    for landmark_num in range(0,68):

        x_point = source_face_landmarks.part(landmark_num).x
        y_point = source_face_landmarks.part(landmark_num).y
        source_face_landmark_points.append((x_point,y_point))
        cv2.circle(source_image,(x_point,y_point),2,(255,0,0),-1)
        cv2.putText(source_image,str(landmark_num),(x_point,y_point),cv2.FONT_HERSHEY_COMPLEX,0.15,(0,255,0))
        #cv2.imshow("1:landmark points of souce",source_image)
    
    #print(source_face_landmark_points)
    ##5.to make convex huull or the outline obtined form joining the oints obtaied in the face landamark array 
    source_face_landmark_points_array = np.array(source_face_landmark_points,np.int32) 
    #print(type(source_face_landmark_points_array),source_face_landmark_points_array.shape)
    source_face_convex_hull = cv2.convexHull(source_face_landmark_points_array)
    cv2.polylines(source_image,[source_face_convex_hull],True,(255,0,0),3)
    #cv2.imshow("2:convex hull of source",source_image)

    ##6.we fill the zero array source canvas with the convexhull shown above 
    cv2.fillConvexPoly(source_image_canvas,source_face_convex_hull,255)
    #cv2.imshow("3:creation of canvas with the mask",source_image_canvas)

    ##7.extract onyl the face
    source_face_image = cv2.bitwise_and(source_image,source_image,mask = source_image_canvas)
    #cv2.imshow("4:joinig of cnvs and source image",source_face_image)

    ##8.delanuay triagnualtion on the osurce iagee
    bounding_rectangle = cv2.boundingRect(source_face_convex_hull)
    subdivisions = cv2.Subdiv2D(bounding_rectangle)
    subdivisions.insert(source_face_landmark_points)
    triangles_vector = subdivisions.getTriangleList()
    triangles_array = np.array(triangles_vector,dtype=np.int32)
    #print("triangles_array: ",triangles_array,triangles_array.shape)
    source_triangle_index_points_list=[]
    for triangle in triangles_array:
        index_pt1 = (triangle[0],triangle[1])
        index_pt2 = (triangle[2],triangle[3])
        index_pt3 = (triangle[4],triangle[5])

        line_color = (0,0,255)
        cv2.line(source_face_image,index_pt1,index_pt2,line_color,1)
        cv2.line(source_face_image,index_pt2,index_pt3,line_color,1)
        cv2.line(source_face_image,index_pt3,index_pt1,line_color,1)

        #cv2.imshow("5:drwaing of all delanuay triangles in the source image",source_face_image)
        #now these delanauy traingels are based upon corners and not the 68 facial landamark points 
        # so we want them to be refernced on basis of 68 facial alndmarks 
        index_pt1=np.where((source_face_landmark_points_array==index_pt1).all(axis=1))
        index_pt1=index_from_array(index_pt1)
        #print(index_pt1)
        index_pt2=np.where((source_face_landmark_points_array==index_pt2).all(axis=1))
        index_pt2=index_from_array(index_pt2)
        #print(index_pt2)
        index_pt3=np.where((source_face_landmark_points_array==index_pt3).all(axis=1))
        index_pt3=index_from_array(index_pt3)
        #print(index_pt3)

        triangle=[index_pt1,index_pt2,index_pt3]
        source_triangle_index_points_list.append(triangle)

    #print(triangle_index_points_list)
    #print("LEN OF TRIANGELS LIST IS ",np.shape(triangle_index_points_list))
    #print(type(triangle_index_points_list))



    ##r9.repeating al the steps for the convex hull fo the destination image
destination_faces = frontal_face_detector(destination_image_grayscale)

for destination_face in destination_faces:    
    destination_face_landmarks = frontal_face_predictor(destination_image_grayscale,destination_face)
    destination_face_landmark_points = []
    for landmark_num in range(0,68):

        x_point = destination_face_landmarks.part(landmark_num).x
        y_point = destination_face_landmarks.part(landmark_num).y
        destination_face_landmark_points.append((x_point,y_point))
        cv2.circle(destination_image,(x_point,y_point),2,(255,0,0),-1)
        cv2.putText(destination_image,str(landmark_num),(x_point,y_point),cv2.FONT_HERSHEY_COMPLEX,0.15,(0,255,0))
        #cv2.imshow("1:landmark points of destination",destination_image)
    
    destination_face_landmark_points_array = np.array(destination_face_landmark_points,np.int32) 
    destination_face_convex_hull = cv2.convexHull(destination_face_landmark_points_array)
    cv2.polylines(destination_image,[destination_face_convex_hull],True,(255,0,0),3)
    #cv2.imshow("2:convex hull of dest image fqace",destination_image)
    #cv2.fillConvexPoly(destination_image_canvas,destination_face_convex_hull,255)

    ##10.for every source traignle fromt he list of triangles crop the bounding rectangle and extract only triangle points
for i,triangle_index_points in enumerate(source_triangle_index_points_list):
    #get x and y coords of the vertices 
    source_triangle_point1 = source_face_landmark_points[triangle_index_points[0]]
    source_triangle_point2 = source_face_landmark_points[triangle_index_points[1]]
    source_triangle_point3 = source_face_landmark_points[triangle_index_points[2]]
    source_triangle = np.array([source_triangle_point1, source_triangle_point2,source_triangle_point3],np.int32)

    source_rectangle = cv2.boundingRect(source_triangle)
    (x,y,w,h)=source_rectangle
    cropped_source_rectangle = source_image[y:y+h,x:x+w]

    source_triangle_points = np.array([[source_triangle_point1[0]-x,source_triangle_point1[1]-y],
                                        [source_triangle_point2[0]-x,source_triangle_point2[1]-y],
                                        [source_triangle_point3[0]-x,source_triangle_point3[1]-y]],np.int32)
    #for as demo slect triagnle 13 and display triagnle lines in white 
    #if i==13:
    #    cv2.line(source_image,source_triangle_point1,source_triangle_point2,255)
    #    cv2.line(source_image,source_triangle_point2,source_triangle_point3,255)
    #    cv2.line(source_image,source_triangle_point3,source_triangle_point1,255)
    #    #cv2.imshow('source triangle lines for i=13',source_image)
    #    cv2.rectangle(source_image,(x,y),(x+w,y+h),(0,0,255),1)
    #    #cv2.imshow("source rect lines",source_image)
    #    #cv2.imshow("cropped_source_rectangle",cropped_source_rectangle)

##11 repeat all these steps of crooping for dest as well
    destination_triangle_point1 = destination_face_landmark_points[triangle_index_points[0]]
    destination_triangle_point2 = destination_face_landmark_points[triangle_index_points[1]]
    destination_triangle_point3 = destination_face_landmark_points[triangle_index_points[2]]
    destination_triangle = np.array([destination_triangle_point1, destination_triangle_point2,destination_triangle_point3],np.int32)

    destination_rectangle = cv2.boundingRect(destination_triangle)
    (x,y,w,h)=destination_rectangle
    cropped_destination_rectangle = source_image[h,w]
    cropped_destination_rectangle_mask = np.zeros((h,w),np.uint8)

    destination_triangle_points = np.array([[destination_triangle_point1[0]-x,destination_triangle_point1[1]-y],
                                        [destination_triangle_point2[0]-x,destination_triangle_point2[1]-y],
                                        [destination_triangle_point3[0]-x,destination_triangle_point3[1]-y]])
    cv2.fillConvexPoly(cropped_destination_rectangle_mask,destination_triangle_points,255)
    #for as demo slect triagnle 13 and display triagnle lines in white 
    #if i==13:
    #    cv2.line(destination_image,destination_triangle_point1,destination_triangle_point2,255)
    #    cv2.line(destination_image,destination_triangle_point2,destination_triangle_point3,255)
    #    cv2.line(destination_image,destination_triangle_point3,destination_triangle_point1,255)
    #    #cv2.imshow('destination triangle lines for i=13',destination_image)
    #    cv2.rectangle(destination_image,(x,y),(x+w,y+h),(0,0,255),1)
    #    #cv2.imshow("destination rect lines",destination_image)
    #    #cv2.imshow("cropped_destination_rectangle mask",cropped_destination_rectangle_mask)
    #    cv2.waitKey(10000)

    ##12now warp the source triangles to match the destination triangles and place dest triagnle mask ove rit 
    #CONVERT TO NP ARRAY
    source_triangle_points = np.float32(source_triangle_points)
    destination_triangle_points = np.float32(destination_triangle_points)
    #CREATETING THE TRANSFORMATION MATRIX For WaRP AFFINE METHOD
    Matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
    #creatng warped triangle 
    warped_triangle = cv2.warpAffine(cropped_source_rectangle,Matrix,(w,h))
    # print(warped_triangle.shape)
    #demo with i=13
    #if i==13:
    #    #cv2.imshow("warped source triangle wrt the destination triangle points",warped_triangle) 
    #    cv2.waitKey(1000)
    #plaacing dest rect mask over the warped triangle 
    warped_triangle= cv2.bitwise_and(warped_triangle,warped_triangle,mask=cropped_destination_rectangle_mask)

    #if i==13:
    #    #cv2.imshow("warped source triangle witht he mask",warped_triangle)
    #    cv2.waitKey(100000)

    ##13 NOW RECONSTRUCTION OF DEST FACE IN A EMPTY CANVAS THE SIZE OF DEST IAMGE
    #destimagcanvas is total 0 of size of dest image
    #cut off white lines in triagnle usking a mask 
    new_dest_face_canvas_area = destination_image_canvas[y:y+h,x:x+w]
    #converting new small csnas to gray 
    new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area,cv2.COLOR_BGR2GRAY)
    #creation of mask to cut the pixels inside ttrinagle exclusing the white lines 
    _,mask_created_triangle  = cv2.threshold(new_dest_face_canvas_area_gray,1,255,cv2.THRESH_BINARY_INV)
    #placing hte mask created 
    warped_triangle = cv2.bitwise_and(warped_triangle,warped_triangle,mask=mask_created_triangle)
    #place te masked triangle indisede the small canvas area 
    new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area,warped_triangle)
    #place the nw canvas with triangle in it to the large dest canvas at the desired location 
    destination_image_canvas[y:y+h,x:x+w] = new_dest_face_canvas_area
    #if i==10:
    #    #cv2.imshow("pasting the triangle at dest canvas",destination_image_canvas)
    #    cv2.waitKey(100000)

#cv2.imshow("completed dest canvas is ",destination_image_canvas)


##now full facesap 
##14 SWAP DEST FACE WITH NEW FACE 

#CUT DEST AFCE AND PASTE THE NEW FACE 
#create a zeros array mask for final iamgr exactly in same size of dest image grayscale 
final_destination_canvas = np.zeros_like(destination_image_grayscale)
final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas,destination_face_convex_hull,255)
#cv2.imshow("final destination face mask",final_destination_face_mask)
final_destination_face_mask = cv2.bitwise_not(final_destination_face_mask)
#to change black col to white and vice versa 
destination_face_masked = cv2.bitwise_and(destination_image,destination_image,mask=final_destination_face_mask)
#cv2.imshow("dest face masked",destination_face_masked)
destination_with_face = cv2.add(destination_face_masked,destination_image_canvas)
cv2.imshow("add face to dest imafe",destination_with_face)

(x, y, w, h) = cv2.boundingRect(destination_face_convex_hull)
destination_face_center_point = (int((x+x+w)/2), int((y+y+h)/2))


# do the seamless clone
seamlesscloned_face = cv2.seamlessClone(destination_with_face, destination_image,
                                        final_destination_face_mask, destination_face_center_point, cv2.NORMAL_CLONE)

cv2.imshow("14: seamlesscloned_face", seamlesscloned_face)

# close all imshow windows when any key is pressed
cv2.waitKey(10000)
cv2.destroyAllWindows()








     
    
