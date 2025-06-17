function result = TriangleTriangleIntersection(tri1, tri2)
    EPSILON = 1e-6;

    % Triangle 1 vertices
    p1 = tri1(1,:);
    p2 = tri1(2,:);
    p3 = tri1(3,:);

    % Triangle 2 vertices
    q1 = tri2(1,:);
    q2 = tri2(2,:);
    q3 = tri2(3,:);

    % Compute edges and normal vectors
    e1 = p2 - p1;
    e2 = p3 - p1;
    h = cross(e2, e1);
    
    % Check if the triangles are parallel
    if dot(h, h) < EPSILON
        result = false;
        return;
    end

    % Compute factors to determine intersection point
    s = (q1 - p1) / h;
    s1 = dot(s, cross(e2, e1));
    s2 = dot(s, cross(e2, q1 - p1));
    t1 = dot(s, cross(e1, q1 - p1));
    t2 = dot(s, cross(e1, e2));

    % Check if the triangles intersect
    if (s1 > 0 && s1 < 1) && (s2 > 0 && s2 < 1) && (t1 > 0 && t1 < 1) && (t2 > 0 && t2 < 1)
        result = true;
    else
        result = false;
    end
end