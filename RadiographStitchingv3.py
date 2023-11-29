#Image stitching algorithm using premade sift
#import stitching #this does not work for me as the stitching algorithm cannot detect features for us
from matplotlib import pyplot as plt
import numpy as np
import cv2, math, os, gc
from deskew import determine_skew
from scipy.optimize import minimize
from scipy import ndimage, stats
import skimage
from datetime import date

""" ****************************************************************************************
#Here is my attempt at an image stitching algorithm
#images fed into the algorithm will have already been preprocessed for noise and rotated
#We know the size of the object, the effective pixel size of camera, and image dimensions
"""
#boolean options before we start anything
AnswerAccepted = False
AnswerAccepted2 = False
NoBlock = False
LeftBlocked = False
RightBlocked = False

#Navigate to the path for the input file which will populate the list InputParameters with a # as the comment delimiter
currentPath = os.path.dirname(os.path.realpath(__file__))
InputFilePath = currentPath+'/StitchInputEDM1_fast.txt'
#InputFilePath = currentPath+'\StitchInputEDM1_June.txt'
#InputFilePath = currentPath+'\StitchInputEDM1_Recent.txt'

InputParameters = []
with open(InputFilePath) as file:
    for line in file:
        line = line.split('#',1)[0]
        line = line.rstrip()
        if line != '':
            InputParameters.append(line)
#print(InputParameters)
InputPath = InputParameters[0]                  #path to where stack of radiographs is located
LeftImgBaseName = InputParameters[1]            #base name of left image e.g. EDM1mm_LeftFiltered_ for EDM1mm_LeftFiltered_0.0_1.tif 
RightImgBaseName = InputParameters[2]           #base name of the right image
OutputBaseName = InputParameters[3]             #what you want output base name to be
OpenBeamName = InputParameters[4]               #open beam file name
startDeg = float(InputParameters[5])            #degree that the image stack starts with (often 0.0)
startinterestDeg = float(InputParameters[6])    #degree that the code should use for mutual information
incDeg = float(InputParameters[7])              #increment between images in the stack
totalImg = float(InputParameters[8])            #number of total images in the stack (usually 360.0 for the tomography stacks)
effectivePixelSize = float(InputParameters[9])  #effective pixel width (mm/pixel) used to verify stitching size
trueObjectSize = float(InputParameters[10])     #size in mm of the actual object (measured/estimated)
OverlapOccluded = InputParameters[11]              #answer to if mutual information is block by beam
SideOccluded = InputParameters[12]              #answer to which side is blocked (N/A if none)
imgnum = startinterestDeg
num = 0

#construct the path to the first image that will be stitched
LeftImgPath = InputPath+LeftImgBaseName+str(imgnum)+'_1.tif'
RightImgPath = InputPath+RightImgBaseName+str(imgnum)+'_1.tif'
OpenBeamPath = InputPath+OpenBeamName
#get todays date so the output path will be dated for the day that it has been done
today = date.today()
OutputPath = InputPath+'Combined_'+str(today)+'/'
if not os.path.exists(OutputPath):
    #path didn't exist so we just made it
    os.makedirs(OutputPath)
else:
    #path did exist so we should either increment or overwrite
    OverWrite = input("Folder Exists: Overwrite? [Y] or [N]\n")
    if OverWrite == 'Y' or OverWrite == "y":
        pass    #Filtered Path stays the same
    elif OverWrite == 'N' or OverWrite == 'n':
        NewFilteredPath = input("Enter a new folder name:\n")
        OutputPath = InputPath+NewFilteredPath+'/'
        #NoisePath = ImagePath+'Noise/'+NewFilteredPath
        os.makedirs(OutputPath)
        #os.makedirs(NoisePath)
    else:
        print('You didnt put Y or N')


#create the array holders for each image that will ultimately be in the loop

LArrayMinFirst = []
LArrayMinSecond = []
RArrayMinFirst = []
RArrayMinSecond = []
LProfile = []
RProfile = []
ShiftIndex = []

#here we start doing things for each image, we typically take the 10 images to use as the basis for stitching, but if the stack less than that, use that instead
while num < min(10,totalImg):
    #open the tif images as np arrays
    LeftOriginal = cv2.imread(LeftImgPath,-1)
    RightOriginal = cv2.imread(RightImgPath,-1)
    if num == 0:
        Limgsize = LeftOriginal.shape
        Rimgsize = RightOriginal.shape
        #camera is a 512x512 pixel array but I could pass in image sizes from cropped images
        LimgWidth = int(Limgsize[0])         
        LimgHeight = int(Limgsize[1])
        RimgWidth = int(Rimgsize[0])
        RimgHeight = int(Rimgsize[1])

    #one thing that needs to be corrected for is if the images are on a different avg gray value
    #this can happen with neutron statistics and noise values causing inconsistent camera readout
    #Want to normalize based off of the open beam seacion of each radiograph
    #camera takes 14-bit image and needs to be visible so I will just scale up the image to better see it as 16bit -> 65536
    #find the maxima so we can scale without exceeding bix value
    #only need to do this for the first image as well as we need the ROI to start but it will be the same for each
    if num == 0:
        BitMax = 65536 #not actually the bit maximum but is close
        Lmax = np.max(LeftOriginal)
        print(Lmax)
        LeftStretched = (LeftOriginal*int(BitMax/Lmax))
        #LeftStretched = (LeftOriginal*BitMax/Lmax)  #i need these to be ints for select ROI
        Rmax = np.max(RightOriginal)
        RightStretched = (RightOriginal*int(BitMax/Rmax))    #i need these to be ints for select ROI

        print('select ROI for left images'' left edge')
        LedgeROI = cv2.selectROI('Select left edge of object with plenty of open beam room',LeftStretched)  #roi output is x,y coord of top left most point, then width is how far to right from that, height how far downc
        print(LedgeROI)
        #if the user types c, this means cancel so we can just put in "expected" ROI 
        if not np.any((LedgeROI)):
            print('contains only zeros, using defaults')
            LedgeROI = np.array([48,162,50,100])   
        #print(LedgeROI)
        print('select ROI for Right image''s right')
        RedgeROI = cv2.selectROI('Select right edge of object with plenty of open beam room', RightStretched)
        if not np.any((RedgeROI)):
            print('Right contains only zeros,edgeng defaults')
            RedgeROI = np.array([200,155,50,100])
    #the order here is weird as we need rows then columns meaning y_start to y_start+height then x_start to x_start+width
    LCroppedSection = LeftOriginal[LedgeROI[1]:(LedgeROI[1]+LedgeROI[3]),LedgeROI[0]:(LedgeROI[0]+LedgeROI[2])]
    RCroppedSection = RightOriginal[RedgeROI[1]:(RedgeROI[1]+RedgeROI[3]),RedgeROI[0]:(RedgeROI[0]+RedgeROI[2])]

    #Now that I have the cropped section where the skew can be seen threshold this to give open and object, only need this once
    if num == 0:
        LThreshVal = skimage.filters.threshold_otsu(LCroppedSection)
        RThreshVal = skimage.filters.threshold_otsu(RCroppedSection)
        print('Threshold Value from OTSU Method: {}'.format(LThreshVal))
    #I dont care to save each one, so for now lets just let this get written over each time
    LCropThresh = LCroppedSection > LThreshVal    
    RCropThresh = RCroppedSection > RThreshVal

    #The Object should be straight up and down so a vertical line would pass through it
    #Test when its goes from True to False 
    i = 0
    j = 0
    LFlipPoint = np.zeros(LedgeROI[3])   #Make a zero array for at what column it goes from True to false
    while i < LedgeROI[3]:  #these are the rows
        while j < LedgeROI[2]:  #these are the columns
            if LCropThresh[i][j] == True:
                j += 1
            else:
                LFlipPoint[i] = j    #first time we hit a False point, record it and break
                break
        j = 0
        i += 1

    #do the same for RHS
    i = 0
    j = 0
    RFlipPoint = np.zeros(RedgeROI[3])   #Make a zero array for at what column it goes from True to false
    while i < RedgeROI[3]:  #these are the rows
        while j < RedgeROI[2]:  #these are the columns
            if RCropThresh[i][j] == False:      #this is opposite as before as assumption is going from obj to air
                j += 1
            else:
                RFlipPoint[i] = j    #first time we hit a False point, record it and break
                #print(RFlipPoint[i])
                break
        j = 0
        i += 1

    #print(FlipPoint)
    Lx = np.arange(0,LedgeROI[3],1)      #make a vector 0 to rows
    Lslope, Lintercept, Lr_value, Lp_value, Lstd_err = stats.linregress(Lx,LFlipPoint)
    print('LHS slope was calculated as {} with r_value of {}'.format(Lslope, Lr_value))
    LFinalY = Lslope*(LedgeROI[3]+1)      #assuming you start at 0 at row 1, then we would finish at Y=slope*numberofrows
    LAngleRad = math.atan(LFinalY/LedgeROI[3])    #get the angle from opp/adj
    LAngleDeg = math.degrees(LAngleRad)

    Rx = np.arange(0,RedgeROI[3],1)      #make a vector 0 to rows
    Rslope, Rintercept, Rr_value, Rp_value, Rstd_err = stats.linregress(Rx,RFlipPoint)
    print('RHS slope was calculated as {} with r_value of {}'.format(Rslope, Rr_value))
    RFinalY = Rslope*(RedgeROI[3]+1)      #assuming you start at 0 at row 1, then we would finish at Y=slope*numberofrows
    RAngleRad = math.atan(RFinalY/RedgeROI[3])    #get the angle from opp/adj
    RAngleDeg = math.degrees(RAngleRad)


    print('Angle needed according to left image is {} degrees and for right image is {}'.format(LAngleDeg,RAngleDeg))
    #now take the average of what the two sides suggest then make that rotation
    AngleDeg = -1*(LAngleDeg+RAngleDeg)/2   
    l_M = cv2.getRotationMatrix2D((LimgWidth/2,LimgHeight/2),AngleDeg,1)
    r_M = cv2.getRotationMatrix2D((RimgWidth/2,RimgHeight/2),AngleDeg,1)
    L_rotated = cv2.warpAffine(LeftOriginal,l_M,(LimgWidth,LimgHeight))
    R_rotated = cv2.warpAffine(RightOriginal,r_M,(RimgWidth,RimgHeight))

    #Here I am taking the ROI for the overlapping information
    if num == 0:
        Lrot_stretched = (L_rotated*int(BitMax/Lmax))
        Rrot_stretched = (R_rotated*int(BitMax/Rmax))
        #The user may need to use only one side or the other of the bore if not both are visible. I will put an option for this
        LOverlapROI = cv2.selectROI('Select Overlapping Region', Lrot_stretched)  #roi output is x,y coord of top left most point, then width is how far to right from that, height how far downc
        #if the user types c, this means cancel so we can just put in "expected" ROI 
        if not np.any((LOverlapROI)):
            print('contains only zeros, using defaults')
            LOverlapROI = np.array([48,162,50,100])   
        #print(LedgeROI)
        print('select ROI for Right image''s right')
        ROverlapROI = cv2.selectROI('Select Overlapping Region', Rrot_stretched)
        if not np.any((ROverlapROI)):
            print('Right contains only zeros,edgeng defaults')
            ROverlapROI = np.array([200,155,50,100])
        #Find the two points where lowest gray value is indicating left and right side of bore
        #to do this, simply find wherever the minimum point is on left half and minimum point on right half
        if LeftBlocked == False:
            LOverlapHalf = int(LOverlapROI[2]/2)      #using the int here makes it such that it must be a whole number as it really doesnt need to be middle
        if RightBlocked == False:
            ROverlapHalf = int(ROverlapROI[2]/2)
        #this is the X axis for Left and Right sides 
        #if NoBlock == True:
        Lroi = np.arange(1,LOverlapROI[2]+1,1)
        Rroi = np.arange(1,ROverlapROI[2]+1,1)
    #the order here is weird as we need rows then columns meaning y_start to y_start+height then x_start to x_start+width
    LOverlapSection = L_rotated[LOverlapROI[1]:(LOverlapROI[1]+LOverlapROI[3]),LOverlapROI[0]:(LOverlapROI[0]+LOverlapROI[2])]
    ROverlapSection = R_rotated[ROverlapROI[1]:(ROverlapROI[1]+ROverlapROI[3]),ROverlapROI[0]:(ROverlapROI[0]+ROverlapROI[2])]
    #Get the average gray value for all rows of first and second half of roi
    #this is the lineout profile for both the left and right hand images
    LOverlapAvg = np.mean(LOverlapSection, axis=0)
    ROverlapAvg = np.mean(ROverlapSection,axis=0)
    LProfile.append(LOverlapAvg)
    RProfile.append(ROverlapAvg)

    #I want to write this out to csv so I can see it for myself
    np.savetxt(OutputPath+'SavedLOverlapAvg_'+str(imgnum)+'.csv',LOverlapAvg)
    np.savetxt(OutputPath+'SavedROverlapAvg_'+str(imgnum)+'.csv',ROverlapAvg)
    #do this for the first ~10 images both left and right to see how consistent these columns are
    #the goal here is to find how the ROI columns relate to each other. 
    #I can find what shift there is between the chosen L and R ROI by doing a sum of square of the residual between them at each shift
    #now I want Lroi to be shifted such that it minimizes sum of squares residual with Rroi
    #I'm going to start by hedging my bet that it is within (+/-) 4 columns, if not, then the code will do further shifts in direction
    if OverlapOccluded == 'Yes' or OverlapOccluded == 'yes' or OverlapOccluded == 'Y' or OverlapOccluded == 'y':
        ijStart = 25    #I want to cut off as mucht from either side as possible so it doesnt mess with the sum square calculations
    else:
        ijStart = 7 #its ok to use the majority of both sides
    shiftArray = [-4,-3,-2,-1,0,1,2,3,4]
    shiftmin = False
    shiftLeft = False
    sumsquares = []
    for shift in shiftArray:
        i = ijStart
        j = ijStart+shift
        inc = 0
        diffArray = np.zeros((max(len(LOverlapAvg),len(ROverlapAvg)),1),dtype=np.uint16)      #Used to be decreased, but doesnt matter if we have empty 0sdecreased by 14 here because the overall profiles are shortened for the shift
        diffSqrArray = np.zeros((max(len(LOverlapAvg),len(ROverlapAvg)),1),dtype=np.uint16)
        while i < (min(len(LOverlapAvg),len(ROverlapAvg))-(ijStart)-5):    #do len-7 because I will be shifting for the other and it needs to be within the confines of the profile
            diffArray[inc] = LOverlapAvg[i] - ROverlapAvg[j]
            #now I need to square this value incase it is negative for when we eventually sum it up
            diffSqrArray[inc] = diffArray[inc]*diffArray[inc]
            i += 1
            j += 1
            inc += 1
        #now that I have gone through the whole profile, I need to compute the sum of these squares
        sumsquares.append(np.sum(diffSqrArray))
        print('The sum of the squares for shift of {} is {}'.format(shift,sumsquares[-1]))   #sumsquares[-1] always takes the last element of the list
        #plt.show()
    #now that we have gone through shifts from -4 to 4, figure out which one of them is the lowest
    print('This is the list of all sum of squares {}'.format(sumsquares))
    MinimumPosition = sumsquares.index(min(sumsquares))
    print('The position of the minimum point is {}'.format(MinimumPosition))

    if MinimumPosition == 0:
        #we have found a spot where the starting shift was best, check the shift of -5
        shift = -5
        shiftArray.append(shift)
        shiftSatisfied = False
        while shiftSatisfied == False:
            i = ijStart
            j = ijStart+shift
            inc = 0
            diffArray = np.zeros((max(len(LOverlapAvg),len(ROverlapAvg)),1),dtype=np.uint16)      #Used to be decreased, but doesnt matter if we have empty 0sdecreased by 14 here because the overall profiles are shortened for the shift
            diffSqrArray = np.zeros((max(len(LOverlapAvg),len(ROverlapAvg)),1),dtype=np.uint16)
            while i < (min(len(LOverlapAvg),len(ROverlapAvg))-ijStart-10):    #do len-7 because I will be shifting for the other and it needs to be within the confines of the profile
                print('Going to {} and i is currently {} while j is currently {}'.format(min(len(LOverlapAvg),len(ROverlapAvg))-ijStart-10,i,j))
                diffArray[inc] = LOverlapAvg[i] - ROverlapAvg[j]
                #now I need to square this value incase it is negative for when we eventually sum it up
                diffSqrArray[inc] = diffArray[inc]*diffArray[inc]
                i += 1
                j += 1
                inc += 1
            #now that I have gone through the whole profile, I need to compute the sum of these squares
            sumsquares.append(np.sum(diffSqrArray))
            print('The sum of the squares for shift of {} is {}'.format(shift,sumsquares[-1]))   #sumsquares[-1] always takes the last element of the list
            if sumsquares[0] > sumsquares[-1]:  #it got better with another shift
                shift = shift-1
                shiftArray.append(shift)
            else:   #it got worst that was the best spot
                shiftSatisfied = True
                MinimumPosition = sumsquares.index(min(sumsquares))

    if MinimumPosition == 8:
        #we have found a spot where the final shift was best, check the shift of 5
        shift = 5
        shiftArray.append(shift)
        shiftSatisfied = False
        while shiftSatisfied == False:
            i = ijStart
            j = ijStart+shift
            inc = 0
            diffArray = np.zeros((max(len(LOverlapAvg),len(ROverlapAvg)),1),dtype=np.uint16)      #Used to be decreased, but doesnt matter if we have empty 0sdecreased by 14 here because the overall profiles are shortened for the shift
            diffSqrArray = np.zeros((max(len(LOverlapAvg),len(ROverlapAvg)),1),dtype=np.uint16)
            while i < (min(len(LOverlapAvg),len(ROverlapAvg))-ijStart-10):    #do len-7 because I will be shifting for the other and it needs to be within the confines of the profile
                print('Going to {} and i is currently {} while j is currently {}'.format(min(len(LOverlapAvg),len(ROverlapAvg))-ijStart-10,i,j))
                diffArray[inc] = LOverlapAvg[i] - ROverlapAvg[j]
                #now I need to square this value incase it is negative for when we eventually sum it up
                diffSqrArray[inc] = diffArray[inc]*diffArray[inc]
                i += 1
                j += 1
                inc += 1
            #now that I have gone through the whole profile, I need to compute the sum of these squares
            sumsquares.append(np.sum(diffSqrArray))
            print('The sum of the squares for shift of {} is {}'.format(shift,sumsquares[-1]))   #sumsquares[-1] always takes the last element of the list
            if sumsquares[-2] > sumsquares[-1]:  #it got better with another shift
                shift = shift+1
                shiftArray.append(shift)
            else:   #it got worst that was the best spot
                shiftSatisfied = True
                MinimumPosition = sumsquares.index(min(sumsquares))
    
    ShiftIndex.append(shiftArray[MinimumPosition])     #I am putting together the list of shifts that it says I need from the 10 images and then the mode of this will be used later
    plt.figure(num)
    plt.plot(Lroi,LProfile[0],label='Left Side')
    plt.plot(Rroi+(-1*shiftArray[MinimumPosition]),RProfile[0],label='Right Side Shifted {}'.format(shiftArray[MinimumPosition]))
    plt.legend()
    plt.title('Overlapping Regions for {} degree at shift {} was selected as best'.format(imgnum,shiftArray[MinimumPosition]))
    #plt.show()
    #if shift is 2 for example, then that means at x = 67 for the Left the closest matching thing is x = 69 for the Right indicating a shift of 2. so the plot needs to be x-2 for the right side for them to match

    del sumsquares

    #iterate numbers
    imgnum = round((imgnum+incDeg),1)
    if imgnum > 360.0:
        imgnum = startDeg       #this is just in case the 10 images of interest overlap other images
    num += 1
    LeftImgPath = InputPath+LeftImgBaseName+str(imgnum)+'_1.tif'
    #LeftImgPath = 'C:/Users/mattg_000/Documents/Research/ImageStitchingAlgorithm/LeftSide/Reordered/LeftSide_'+str(imgnum)+'_1.tif'
    #print(LeftImgPath)
    RightImgPath = InputPath+RightImgBaseName+str(imgnum)+'_1.tif'
    #delete the items that will be written over again
    del LeftOriginal
    del RightOriginal
    del L_rotated
    del R_rotated

plt.show()  #shows all of the shifted plots after it has gone through them

#now that we have the list of all of the shifts for the 10, find whatever shift was most likely (aka mode)
ShiftMed = int(np.median(ShiftIndex))
print('The final shift array is {} and has a median of {}'.format(ShiftIndex,ShiftMed))
#Now that we have that, I know that ShiftMed is how much I need to shift the ROI columns from Right side to match left side
#aka from starting column of LOverlapROI (which is LOverlapROI[0]) to ending (which is LOverlapROI[0]+LOverlapROI[2]) those are the ones that overlap with
#starting column of shifted ROverlapROI-ShiftMed (which is ROverlapROI[0]-ShiftMed) to ending (which is ROverlapROI[0]+ROverlapROI[2]-ShiftMed)
#now actually I need to decide which width to use for the ROI, I'll just pick whichever is smaller so it is all the same
OverlapWidth = min(LOverlapROI[2],ROverlapROI[2])
LOverlapStart = LOverlapROI[0]
LOverlapEnd = LOverlapROI[0]+OverlapWidth
ROverlapStart = ROverlapROI[0]-ShiftMed
ROverlapEnd = ROverlapROI[0]+OverlapWidth-ShiftMed

#This will be a logistic function as it should be mostly left side until we get closer to middle and same for oppposite
#if the the feature of interest "Bore" is blocked by the beam at all, I will shift the array to favor the side that is not blocked
xarray = np.arange(0,OverlapWidth+1,1)
weightlogistic = np.zeros((len(xarray),1))
if OverlapOccluded == 'n' or OverlapOccluded == 'no' or OverlapOccluded == 'N' or OverlapOccluded == 'No':
    xmed = np.median(xarray)
elif OverlapOccluded == 'y' or OverlapOccluded == 'yes' or OverlapOccluded == 'Y' or OverlapOccluded == 'Yes':
    if SideOccluded == 'Left' or SideOccluded == 'left' or SideOccluded == 'L' or SideOccluded == 'l' or SideOccluded == 'Bottom' or SideOccluded == 'bottom' or SideOccluded == 'B' or SideOccluded == 'b':
        xmed = np.quantile(xarray,0.25)
    elif SideOccluded == 'Right' or SideOccluded == 'right' or SideOccluded == 'R' or SideOccluded == 'r' or SideOccluded == 'Top' or SideOccluded == 'top' or SideOccluded == 'T' or SideOccluded == 't':
        xmed = np.quantile(xarray,0.75)
    elif SideOccluded == 'Both' or 'both':
        xmed = np.median(xarray)
    else:
        print('User input incorrect format for which side is occluded using just the middle value')
        xmed = np.median(xarray)
else:
    print('User input incorrect format for if the feature overlapping feature is occluded by either side of the beam')
    xmed = np.median(xarray)
inc = 0
k = 0.2
for x in xarray:
    weightlogistic[inc] = 1/(1+np.exp(-k*(x-xmed)))
    inc += 1

plt.figure(11)
plt.plot(xarray,weightlogistic)
plt.xlabel('Position')
plt.ylabel('Normalized Weight')
plt.title('Weighting function to blend overlapping ROI')
plt.show()

imgnum2 = startDeg
num2 = 0
SideNum = 0
CropRoiSelected = False
EdgeRoiSelected = False
roiSelected = False
LeftImgPath = InputPath+LeftImgBaseName+str(imgnum2)+'_1.tif'
RightImgPath = InputPath+RightImgBaseName+str(imgnum2)+'_1.tif'

#now I want to make combined images for a few pixels to left and right of the "expected"
combinedpath = OutputPath
#combinedpath = 'C:/Users/mattg_000/Documents/Research/ImageStitchingAlgorithm/CombinedBlended'
while num2 < totalImg:  #360:
    #open the tif images as np arrays
    #starts off doing the same thing as last loop as now we need to do these functions again then eventaully write new image
    LeftOriginal2 = cv2.imread(LeftImgPath,-1)
    RightOriginal2 = cv2.imread(RightImgPath,-1)
    LCroppedSection2 = LeftOriginal2[LedgeROI[1]:(LedgeROI[1]+LedgeROI[3]),LedgeROI[0]:(LedgeROI[0]+LedgeROI[2])]
    RCroppedSection2 = RightOriginal2[RedgeROI[1]:(RedgeROI[1]+RedgeROI[3]),RedgeROI[0]:(RedgeROI[0]+RedgeROI[2])]

    LCropThresh = LCroppedSection2 > LThreshVal    
    RCropThresh = RCroppedSection2 > RThreshVal
    #print(LCropThresh)
    #The Object should be straight up and down so a vertical line would pass through it
    #Test when its goes from True to False 
    i = 0
    j = 0
    LFlipPoint = np.zeros(LedgeROI[3])   #Make a zero array for at what column it goes from True to false
    while i < LedgeROI[3]:  #these are the rows
        while j < LedgeROI[2]:  #these are the columns
            if LCropThresh[i][j] == True:
                j += 1
            else:
                LFlipPoint[i] = j    #first time we hit a False point, record it and break
                #print(LFlipPoint[i])
                break
        j = 0
        i += 1

    #do the same for RHS
    i = 0
    j = 0
    RFlipPoint = np.zeros(RedgeROI[3])   #Make a zero array for at what column it goes from True to false
    while i < RedgeROI[3]:  #these are the rows
        while j < RedgeROI[2]:  #these are the columns
            if RCropThresh[i][j] == False:      #this is opposite as before as assumption is going from obj to air
                j += 1
            else:
                RFlipPoint[i] = j    #first time we hit a False point, record it and break
                #print(RFlipPoint[i])
                break
        j = 0
        i += 1

    #print(FlipPoint)
    Lx = np.arange(0,LedgeROI[3],1)      #make a vector 0 to rows
    Lslope, Lintercept, Lr_value, Lp_value, Lstd_err = stats.linregress(Lx,LFlipPoint)
    #print('LHS slope was calculated as {} with r_value of {}'.format(Lslope, Lr_value))
    #plt.imshow(LCropThresh)
    LFinalY = Lslope*(LedgeROI[3]+1)      #assuming you start at 0 at row 1, then we would finish at Y=slope*numberofrows
    LAngleRad = math.atan(LFinalY/LedgeROI[3])    #get the angle from opp/adj
    #LAngleRad = math.atan(Lslope)
    LAngleDeg = math.degrees(LAngleRad)

    Rx = np.arange(0,RedgeROI[3],1)      #make a vector 0 to rows
    Rslope, Rintercept, Rr_value, Rp_value, Rstd_err = stats.linregress(Rx,RFlipPoint)
    #print('RHS slope was calculated as {} with r_value of {}'.format(Rslope, Rr_value))
    #plt.imshow(LCropThresh)
    RFinalY = Rslope*(RedgeROI[3]+1)      #assuming you start at 0 at row 1, then we would finish at Y=slope*numberofrows
    RAngleRad = math.atan(RFinalY/RedgeROI[3])    #get the angle from opp/adj
    #RAngleRad = math.atan(Rslope)
    RAngleDeg = math.degrees(RAngleRad)

    #now take the average of what the two sides suggest then make that rotation
    AngleDeg = -1*(LAngleDeg+RAngleDeg)/2   
    l_M = cv2.getRotationMatrix2D((LimgWidth/2,LimgHeight/2),AngleDeg,1)
    r_M = cv2.getRotationMatrix2D((RimgWidth/2,RimgHeight/2),AngleDeg,1)
    L_rotated2 = cv2.warpAffine(LeftOriginal2,l_M,(LimgWidth,LimgHeight))
    R_rotated2 = cv2.warpAffine(RightOriginal2,r_M,(RimgWidth,RimgHeight))
    
    #Now I need to make sure that each object after being rotated is lined up so we will find the row that we switch from open beam to object for both
    if roiSelected == False:
        LTopRoi = cv2.selectROI('Select Top section with plenty of open beam', (L_rotated2*int(BitMax/(Lmax*1.5))))
        RTopRoi = cv2.selectROI('Select Top section with plenty of open beam', (R_rotated2*int(BitMax/(Rmax*1.5))))
        roiSelected = True

    LTopCroppedSection = L_rotated2[LTopRoi[1]:(LTopRoi[1]+LTopRoi[3]),LTopRoi[0]:(LTopRoi[0]+LTopRoi[2])]
    RTopCroppedSection = R_rotated2[RTopRoi[1]:(RTopRoi[1]+RTopRoi[3]),RTopRoi[0]:(RTopRoi[0]+RTopRoi[2])]
    #Now that I have the cropped section where the skew can be seen threshold this to give open and object, only need this once
    if num2 == 0:
        LTopThreshVal = skimage.filters.threshold_otsu(LTopCroppedSection)
        RTopThreshVal = skimage.filters.threshold_otsu(RTopCroppedSection)
        print('Threshold Value from OTSU Method: {}'.format(LTopThreshVal))
    #I dont care to save each one, so for now lets just let this get written over each time
    LTopCropThresh = LTopCroppedSection > LTopThreshVal    
    RTopCropThresh = RTopCroppedSection > RTopThreshVal
    #print(LCropThresh)
    #The Object should be straight up and down so a horizontal line would pass through it
    #Test when its goes from True to False 
    i = 0
    j = 0
    LTopFlipPoint = np.zeros(LTopRoi[2])   #Make a zero array for at what row it goes from True to false
    while j < LTopRoi[2]:  #these are the columns aka width
        while i < LTopRoi[3]:  #these are the rows aka height
            if LTopCropThresh[i][j] == True:
                i += 1
            else:
                LTopFlipPoint[j] = i    #first time we hit a False point, record it and break
                #print(LFlipPoint[i])
                break
        i = 0
        j += 1

    #do the same for RHS
    i = 0
    j = 0
    RTopFlipPoint = np.zeros(RTopRoi[2])   #Make a zero array for at what column it goes from True to false
    while j < RTopRoi[2]:  #these are the columns aka width
        while i < RTopRoi[3]:  #these are the rows aka height
            if RTopCropThresh[i][j] == True:      #air
                i += 1
            else:
                RTopFlipPoint[j] = i    #first time we hit a False point, record it and break
                #print(RFlipPoint[i])
                break
        i = 0
        j += 1

    #now we need both of these flip points to be at the same spot, so look at the median flip point
    LTopMedFlip = int(np.median(LTopFlipPoint))
    RTopMedFlip = int(np.median(RTopFlipPoint))
    LFlipRow = LTopMedFlip+LTopRoi[1]
    RFlipRow = RTopMedFlip+RTopRoi[1]
    if LFlipRow != RFlipRow:  #if these aren't at the same spot, we need to translate one side up (lets just do right side)
        ty = LFlipRow-RFlipRow
        print('The Left flip row was {}, right flip row as {} and that meant a translation of {} was needed'.format(LFlipRow,RFlipRow,ty))
        #ty = LTopMedFlip - RTopMedFlip
        #print('Needs to be shifted by {} rows'.format(ty))
        Rshift = np.float32([[1, 0, 0],[0, 1, ty]])   #this is a 2x3 matrix identity for first 2x2 then third column, we get a shift in y from ty
        R_shifted = cv2.warpAffine(R_rotated2,Rshift,(RimgWidth,RimgHeight))
    else:
        R_shifted = R_rotated2
    
    if num2 == 0:
        Lrot_stretched = (L_rotated2*int(BitMax/Lmax))
        Rrot_stretched = (R_rotated2*int(BitMax/Rmax))  
        LOpenROI = cv2.selectROI('Select Open Beam Region', Lrot_stretched)
        if not np.any((LOpenROI)):
            print('contains only zeros, using defaults')
            LOpenROI = np.array([48,162,50,100])   
        #print(LedgeROI)
        print('select ROI for Right image''s right')
        ROpenROI = cv2.selectROI('Select Open Beam Region', Rrot_stretched)
        if not np.any((ROpenROI)):
            print('Right contains only zeros,edgeng defaults')
            ROpenROI = np.array([200,155,50,100])
    #I want to have a new shift for each image in case there is a difference between images that changes with time (aka camera hasnt warmed up)
    meanL = np.mean(LeftOriginal2[LOpenROI[1]:(LOpenROI[1]+LOpenROI[3]),LOpenROI[0]:(LOpenROI[0]+LOpenROI[2])])
    meanR = np.mean(RightOriginal2[ROpenROI[1]:(ROpenROI[1]+ROpenROI[3]),ROpenROI[0]:(ROpenROI[0]+ROpenROI[2])])
    meanBoth = (meanL+meanR)/2
    Lmult = meanL/meanBoth  #multiplier for the left side
    Rmult = meanR/meanBoth  #multiplier for the right side
    #create a corrected image from the two originals
    
    L_intensityCorrected2 = np.array((L_rotated2/Lmult), dtype=np.uint16)       #cast all values after correcting to uint16 as tiffs need that to be saved
    R_intensityCorrected2 = np.array((R_shifted/Rmult), dtype=np.uint16)
    #R_intensityCorrected2 = np.array((R_rotated2/Rmult), dtype=np.uint16)

    #Now we combine the two arrays together L ending at LeftMiddle then picking up at RightMiddle

    #this is how it was originally, now that I'm looping it looks different
    LeftFull = L_intensityCorrected2[:,0:LOverlapStart] #was overlapStart-1 but I think I need it to be up to the whole thing cuz of how numpy slicing works
    #now I need to do the blended for the Left and Right Where they overlap
    blendInc = 0
    leftInc = LOverlapStart
    rightInc = ROverlapStart
    BlendArray = np.zeros((LimgHeight,OverlapWidth))#,dtype = np.uint16)
    while blendInc < OverlapWidth:
        BlendArray[:,blendInc] = (1-weightlogistic[blendInc])*L_intensityCorrected2[:,leftInc]+weightlogistic[blendInc]*R_intensityCorrected2[:,rightInc]        
        blendInc += 1
        leftInc += 1
        rightInc += 1
        #weightFactor = weightFactor - weightShift
    #Now that I have made the blended array I need to make sure everything is put as an integer
    BlendArray = BlendArray.astype(np.uint16)
    RightFull = R_intensityCorrected2[:,ROverlapEnd+1:]
    if num2 == 0:
        print('First thing is the last 250th and 251 row of last column for left\n')
        print(LeftFull[250,-1],LeftFull[251,-1])
        print('Second one is the first column of the blended array (to make sure it isnt the same) where weighting here is {}\n'.format(1-weightlogistic[0]))
        print(BlendArray[250,0],BlendArray[251,0])
        print('Next thing is the last column of the blend array\n')
        print(BlendArray[250,-1],BlendArray[251,-1])
        print('And finally the first column of right full')
        print(RightFull[250,0],RightFull[251,1])

    #print('Shape of Left {} and shape of right {}'.format(LeftHalf.shape,RightHalf.shape))
    Combine = np.concatenate((LeftFull,BlendArray,RightFull),axis=1)   #axis=1 means we are combining along vertical axis
    #Now pull up the image to crop it down to final size
    if EdgeRoiSelected == False:
        Combstretched = (Combine*int(BitMax/(Lmax*1.5)))
        #check the combined image to see if the size of the combined object matches measured size
        LeftEdgeFromCombinedROI = cv2.selectROI('Select the left edge of combined with plenty of open beam room',Combstretched)
        if not np.any(LeftEdgeFromCombinedROI):
            print('Contains only zeros, using defaults')
            LeftEdgeFromCombinedROI = np.array([48,162,50,100])
        RightEdgeFromCombinedROI = cv2.selectROI('Select the right edge of combined with plenty of open beam room', Combstretched)
        if not np.any((RightEdgeFromCombinedROI)):
            print('Right contains only zeros,edge defaults')
            RedgeROI = np.array([400,155,50,100])
        EdgeRoiSelected = True
        #the order here is weird as we need rows then columns meaning y_start to y_start+height then x_start to x_start+width
        LeftCombinedSection = Combine[LeftEdgeFromCombinedROI[1]:(LeftEdgeFromCombinedROI[1]+LeftEdgeFromCombinedROI[3]),LeftEdgeFromCombinedROI[0]:(LeftEdgeFromCombinedROI[0]+LeftEdgeFromCombinedROI[2])]
        RightCombinedSection = Combine[RightEdgeFromCombinedROI[1]:(RightEdgeFromCombinedROI[1]+RightEdgeFromCombinedROI[3]),RightEdgeFromCombinedROI[0]:(RightEdgeFromCombinedROI[0]+RightEdgeFromCombinedROI[2])]
        #Now that I have the cropped section where the skew can be seen threshold this to give open and object, only need this once
        LCombinedThreshVal = skimage.filters.threshold_otsu(LeftCombinedSection)
        RCombinedThreshVal = skimage.filters.threshold_otsu(RightCombinedSection)
        print('Threshold Value from OTSU Method: {}'.format(LCombinedThreshVal))
        #I dont care to save each one, so for now lets just let this get written over each time
        LCombinedThresh = LeftCombinedSection > LCombinedThreshVal    
        RCombinedThresh = RightCombinedSection > RCombinedThreshVal
        #print(LCropThresh)
        #The Object should be straight up and down so a vertical line would pass through it
        #Test when its goes from True to False 
        i = 0
        j = 0
        LCombinedFlipPoint = np.zeros(LeftEdgeFromCombinedROI[3])   #Make a zero array for at what column it goes from True to false
        while i < LeftEdgeFromCombinedROI[3]:  #these are the rows
            while j < LeftEdgeFromCombinedROI[2]:  #these are the columns
                if LCombinedThresh[i][j] == True:
                    j += 1
                else:
                    LCombinedFlipPoint[i] = j    #first time we hit a False point, record it and break
                    #print(LFlipPoint[i])
                    break
            j = 0
            i += 1

        #do the same for RHS
        i = 0
        j = 0
        RCombinedFlipPoint = np.zeros(RightEdgeFromCombinedROI[3])   #Make a zero array for at what column it goes from True to false
        while i < RightEdgeFromCombinedROI[3]:  #these are the rows
            while j < RightEdgeFromCombinedROI[2]:  #these are the columns
                if RCombinedThresh[i][j] == False:      #this is opposite as before as assumption is going from obj to air
                    j += 1
                else:
                    RCombinedFlipPoint[i] = j    #first time we hit a False point, record it and break
                    #print(RFlipPoint[i])
                    break
            j = 0
            i += 1
        
        #now I want to get the mean flip point value for left and right side
        LCombinedFlipMean = np.mean(LCombinedFlipPoint) #this is the point from 0-width of ROI so I need to add LeftEdgeFromCombined[0] (x-coordinate of left edge)
        RCombinedFlipMean = np.mean(RCombinedFlipPoint)
        LCombinedFlipColumn = int(LCombinedFlipMean)+LeftEdgeFromCombinedROI[0]
        RCombinedFlipColumn = int(RCombinedFlipMean)+RightEdgeFromCombinedROI[0]
        print('The average flip column on left side is {} and right side is {}'.format(LCombinedFlipColumn,RCombinedFlipColumn))
        OutputObjectColumns = RCombinedFlipColumn-LCombinedFlipColumn
        OutputObjectSize = OutputObjectColumns*effectivePixelSize   #translate columns into mm by using mm/pixel from pixel size
        #now compare the object size in mm between known object size. If they are not close enough we have a failed stitch
        #im going to start of with it needing to be within 2% of the true object size
        print('The estimated object size is {}'.format(OutputObjectSize))
        if np.abs(trueObjectSize-OutputObjectSize) > 0.02*trueObjectSize:
            print('Stitching algorithm stitched at the incorrect, the output object size is too far from true size')
            raise Exception("The stitch algorithm failed due to improper sizing of the final image")

    #this is the step to crop the image down to the final size
    if CropRoiSelected == False:
        CombCropROI = cv2.selectROI('Crop the stitched image down to the final desired size',Combstretched)  #roi output is x,y coord of top left most point, then width is how far to right from that, height how far downc
        print('Region of Interest for Final Cropping:')
        print(CombCropROI)
        #if the user types c, this means cancel so we can just put in "expected" ROI 
        if not np.any((CombCropROI)):
            print('contains only zeros, using defaults')
            CombCropROI = np.array([48,162,50,100])
        CropRoiSelected = True  #now that we've gotten the ROIs we need, we can never look at this again

    #this is the step for the open beam image
    if num2 == 0:
        #we also need to do the same rotations and corretions for the open beam image
        OpenImage = cv2.imread(OpenBeamPath,-1)
        OpenRot = cv2.warpAffine(OpenImage,l_M,(LimgWidth,LimgHeight))
        OpenLeft = OpenImage[:,0:LOverlapStart-1]        #OpenImage[:,0:LeftMiddle]
        blendInc = 0
        leftInc = LOverlapStart
        rightInc = ROverlapStart
        BlendArray = np.zeros((LimgHeight,OverlapWidth))#,dtype = np.uint16)
        weightFactor = 1.0  #weight factor is how much each array contributes so it starts at 1.0 meaning on the left side of the two arrays all of Left is in and none of right. but it will shift
        weightShift = 1/(OverlapWidth-1)    #this gives how much that weighting changes based on the size of the overlap
        while blendInc < OverlapWidth:
            BlendArray[:,blendInc] = weightFactor*OpenImage[:,leftInc]+(1-weightFactor)*OpenImage[:,rightInc]        
            blendInc += 1
            leftInc += 1
            rightInc += 1
            weightFactor = weightFactor - weightShift
        #Now that I have made the blended array I need to make sure everything is put as an integer
        BlendArray = BlendArray.astype(np.uint16)
        OpenRight = OpenImage[:,ROverlapEnd+1:]               #OpenImage[:,RightMiddle:]
        OpenCombine = np.concatenate((OpenLeft,BlendArray,OpenRight),axis=1)
        OpenCombineCropped = OpenCombine[CombCropROI[1]:(CombCropROI[1]+CombCropROI[3]),CombCropROI[0]:(CombCropROI[0]+CombCropROI[2])]
        #cv2.imwrite('C:/Users/mattg_000/Documents/Research/ImageStitchingAlgorithm/OpenCombined2023_Blended.tif',OpenCombineCropped)
        cv2.imwrite(OutputPath+'OpenCombined_Blended.tif',OpenCombineCropped)
    CombineCropped = Combine[CombCropROI[1]:(CombCropROI[1]+CombCropROI[3]),CombCropROI[0]:(CombCropROI[0]+CombCropROI[2])]
    #save the combined image
    CombinedImagePath = OutputPath
    #make the directory if it doesn't exist
    if os.path.exists(CombinedImagePath) == False:
        os.mkdir(CombinedImagePath)
    CombinedImageName = CombinedImagePath+OutputBaseName+str(num2)+'.tif'
    cv2.imwrite(CombinedImageName,CombineCropped)

    imgnum2 = round((imgnum2+incDeg),1)
    num2 += 1 

    LeftImgPath = InputPath+LeftImgBaseName+str(imgnum2)+'_1.tif'
    RightImgPath = InputPath+RightImgBaseName+str(imgnum2)+'_1.tif'

    del LeftOriginal2
    del RightOriginal2
    del L_intensityCorrected2
    del R_intensityCorrected2
    del L_rotated2
    del R_rotated2
    del R_shifted
    gc.collect()
SideNum += 1
num2 = 0
imgnum2 = startDeg
