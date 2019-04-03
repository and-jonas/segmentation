#####insert
np.argsort(contours, -stats[:, -1])

# Finding contours for the thresholded image
im2, contours, hierarchy = cv2.findContours(mask_cleaned_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im2, contours, hierarchy = cv2.findContours(mask_cleaned_3, 2, 1)

for i in:

cnt = contours[11]
cv2.moments(cnt)

hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(Color_Masked, start, end, [0, 255, 0], 2)
    cv2.circle(Color_Masked, far, 5, [0, 0, 255], -1)

# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(Color_Masked)
ax.set_title("img")
plt.tight_layout()
plt.show()

# Create hull array for convex hull points
hull = []
# Calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

defects = []
for i in len(hull):
    defects.append(cv2.convexityDefects(contours[i], hull[[i]]))

# Draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(Color_Masked, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(Color_Masked, hull, i, color, 1, 8)

#####insert

# Find right-most contour
maxx = []
for i in range(0, len(contours)):
    maxx.append(contours[i][0:, 0].max())

cnt_right = contours[maxx.index(max(maxx))]

hull = cv2.convexHull(cnt_right, returnPoints=False)
defects = cv2.convexityDefects(cnt_right, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt_right[s][0])
    end = tuple(cnt_right[e][0])
    far = tuple(cnt_right[f][0])
    cv2.line(Color_Masked, start, end, [0, 255, 0], 2)
    if d > 50000:
        cv2.circle(Color_Masked, far, 5, [0, 0, 255], -1)
    cv2.drawContours(Color_Masked, cnt_right, -1, (128, 255, 0), 2)

# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(Color_Masked)
ax.set_title("img")
plt.tight_layout()
plt.show()
